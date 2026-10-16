// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package anthropicmodel

import (
	"context"
	"fmt"
	"iter"
	"log/slog"
	"os"
	"runtime"
	"slices"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"golang.org/x/oauth2/google"

	"google.golang.org/adk/v2/internal/version"
	"google.golang.org/adk/v2/model"
)

const cloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

type anthropicModel struct {
	name        string
	modelID     string
	backend     Backend
	project     string
	location    string
	maxTokens   int
	effort      Effort
	promptCache *PromptCacheConfig
	cfgLogger   *slog.Logger
	client      anthropic.Client
}

// NewModel creates a new [model.LLM] backed by Anthropic Claude on either the
// first-party Anthropic API ([BackendAnthropic]) or Google Cloud Vertex AI
// ([BackendVertex]).
func NewModel(modelName string, cfg ClientConfig) (model.LLM, error) {
	resProject, resLocation, modelID, isResource, err := parseModelName(modelName)
	if err != nil {
		return nil, err
	}

	backend := cfg.Backend
	if backend == "" {
		backend = BackendAnthropic
	}
	if backend != BackendAnthropic && backend != BackendVertex {
		return nil, fmt.Errorf("%w: unsupported Backend %q; valid values are %q, %q", ErrInvalidConfig, backend, BackendAnthropic, BackendVertex)
	}

	if cfg.MaxTokens < 0 {
		return nil, fmt.Errorf("%w: MaxTokens must be non-negative", ErrInvalidConfig)
	}
	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = DefaultMaxTokens
	}

	if cfg.MaxRetries != nil && *cfg.MaxRetries < 0 {
		return nil, fmt.Errorf("%w: MaxRetries must be non-negative", ErrInvalidConfig)
	}

	if err := validateEffort(cfg.Effort); err != nil {
		return nil, err
	}
	if err := validatePromptCacheConfig(cfg.PromptCache); err != nil {
		return nil, err
	}

	if cfg.BaseURL != "" {
		if err := validateBaseURL(cfg.BaseURL); err != nil {
			return nil, err
		}
	}

	var (
		opts     []option.RequestOption
		project  string
		location string
	)

	// Always disable the SDK's ambient environment autoload so we control
	// credential precedence and the custom BaseURL credential guard.
	opts = append(opts, option.WithoutEnvironmentDefaults())

	switch backend {
	case BackendAnthropic:
		if isResource {
			return nil, fmt.Errorf("%w: Vertex resource model name requires BackendVertex", ErrInvalidConfig)
		}
		apiKey := strings.TrimSpace(cfg.APIKey)
		authToken := strings.TrimSpace(cfg.AuthToken)

		if cfg.BaseURL != "" {
			if apiKey == "" && authToken == "" {
				return nil, fmt.Errorf("%w: custom BaseURL on BackendAnthropic requires an explicit APIKey or AuthToken in ClientConfig; ambient environment credentials are not sent to custom endpoints", ErrInvalidConfig)
			}
		} else if apiKey == "" && authToken == "" {
			if envKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); envKey != "" {
				apiKey = envKey
			} else if envToken := strings.TrimSpace(os.Getenv("ANTHROPIC_AUTH_TOKEN")); envToken != "" {
				authToken = envToken
			}
		}

		if apiKey == "" && authToken == "" {
			return nil, fmt.Errorf("%w: %w: no Anthropic credential found; set ClientConfig.APIKey, ClientConfig.AuthToken, ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN", ErrInvalidConfig, ErrAuthentication)
		}

		if cfg.HTTPClient != nil {
			opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
		}
		if cfg.BaseURL != "" {
			opts = append(opts, option.WithBaseURL(strings.TrimSpace(cfg.BaseURL)))
		}
		if apiKey != "" {
			opts = append(opts, option.WithAPIKey(apiKey))
		}
		if authToken != "" {
			opts = append(opts, option.WithAuthToken(authToken))
		}

	case BackendVertex:
		cfgProject := strings.TrimSpace(cfg.Project)
		cfgLocation := strings.TrimSpace(cfg.Location)

		if isResource {
			if cfgProject != "" && cfgProject != resProject {
				return nil, fmt.Errorf("%w: ClientConfig.Project %q does not match resource project %q", ErrInvalidConfig, cfgProject, resProject)
			}
			if cfgLocation != "" && cfgLocation != resLocation {
				return nil, fmt.Errorf("%w: ClientConfig.Location %q does not match resource location %q", ErrInvalidConfig, cfgLocation, resLocation)
			}
		}

		project = firstNonEmpty(
			cfgProject,
			resProject,
			strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_PROJECT")),
			strings.TrimSpace(os.Getenv("CLOUD_ML_PROJECT_ID")),
		)
		location = firstNonEmpty(
			cfgLocation,
			resLocation,
			strings.TrimSpace(os.Getenv("GOOGLE_CLOUD_LOCATION")),
			strings.TrimSpace(os.Getenv("CLOUD_ML_REGION")),
		)

		if project == "" {
			return nil, fmt.Errorf("%w: BackendVertex requires a project via ClientConfig.Project, a full resource model name, GOOGLE_CLOUD_PROJECT, or CLOUD_ML_PROJECT_ID", ErrInvalidConfig)
		}
		if location == "" {
			return nil, fmt.Errorf("%w: BackendVertex requires a location via ClientConfig.Location, a full resource model name, GOOGLE_CLOUD_LOCATION, or CLOUD_ML_REGION", ErrInvalidConfig)
		}

		if cfg.BaseURL != "" && cfg.Credentials == nil && cfg.TokenSource == nil {
			return nil, fmt.Errorf("%w: custom BaseURL on BackendVertex requires explicit Credentials or TokenSource in ClientConfig; Application Default Credentials are not sent to custom endpoints", ErrInvalidConfig)
		}

		creds, err := resolveVertexCredentials(project, cfg)
		if err != nil {
			return nil, err
		}

		// WithHTTPClient must be registered before vertex.WithCredentials so the
		// Vertex option wraps the custom HTTPClient's transport with OAuth2.
		if cfg.HTTPClient != nil {
			opts = append(opts, option.WithHTTPClient(cfg.HTTPClient))
		}

		vertexOpt, err := buildVertexOption(location, project, creds)
		if err != nil {
			return nil, err
		}
		opts = append(opts, vertexOpt)

		if cfg.BaseURL != "" {
			opts = append(opts, option.WithBaseURL(strings.TrimSpace(cfg.BaseURL)))
		}

		trackingHeader := fmt.Sprintf("google-adk/%s gl-go/%s", version.Version, strings.TrimPrefix(runtime.Version(), "go"))
		opts = append(opts,
			option.WithHeader("x-goog-api-client", trackingHeader),
			option.WithHeader("user-agent", trackingHeader),
		)
	}

	if cfg.MaxRetries != nil {
		opts = append(opts, option.WithMaxRetries(*cfg.MaxRetries))
	}
	if len(cfg.RequestOptions) > 0 {
		opts = append(opts, cfg.RequestOptions...)
	}

	var promptCache *PromptCacheConfig
	if cfg.PromptCache != nil {
		copied := *cfg.PromptCache
		if copied.TTL == "" {
			copied.TTL = PromptCacheTTL5m
		}
		promptCache = &copied
	}

	return &anthropicModel{
		name:        strings.TrimSpace(modelName),
		modelID:     modelID,
		backend:     backend,
		project:     project,
		location:    location,
		maxTokens:   maxTokens,
		effort:      cfg.Effort,
		promptCache: promptCache,
		cfgLogger:   cfg.Logger,
		client:      anthropic.NewClient(opts...),
	}, nil
}

func resolveVertexCredentials(project string, cfg ClientConfig) (*google.Credentials, error) {
	if cfg.Credentials != nil {
		if cfg.Credentials.TokenSource == nil {
			return nil, fmt.Errorf("%w: ClientConfig.Credentials must have a non-nil TokenSource", ErrInvalidConfig)
		}
		return cfg.Credentials, nil
	}
	if cfg.TokenSource != nil {
		return &google.Credentials{
			ProjectID:   project,
			TokenSource: cfg.TokenSource,
		}, nil
	}
	creds, err := google.FindDefaultCredentials(context.Background(), cloudPlatformScope)
	if err != nil {
		return nil, fmt.Errorf("%w: %w: failed to find Google Application Default Credentials for Vertex AI", ErrInvalidConfig, ErrAuthentication)
	}
	return creds, nil
}

func buildVertexOption(location, project string, creds *google.Credentials) (opt option.RequestOption, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%w: failed to initialize Vertex AI client option: %v", ErrInvalidConfig, r)
		}
	}()
	return vertex.WithCredentials(context.Background(), location, project, creds), nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// Name returns the model name passed to [NewModel].
func (m *anthropicModel) Name() string {
	return m.name
}

func (m *anthropicModel) logger() *slog.Logger {
	if m.cfgLogger != nil {
		return m.cfgLogger
	}
	return slog.Default() //nolint:forbidigo // unsupported content is skipped with a warning rather than failing the request
}

func (m *anthropicModel) resolveModelID(reqModel string) (string, error) {
	trimmed := strings.TrimSpace(reqModel)
	if trimmed == "" {
		return m.modelID, nil
	}
	resProject, resLocation, modelID, isResource, err := parseModelName(trimmed)
	if err != nil {
		return "", err
	}
	if isResource {
		if m.backend != BackendVertex {
			return "", fmt.Errorf("%w: Vertex resource model name requires BackendVertex", ErrInvalidConfig)
		}
		if resProject != m.project {
			return "", fmt.Errorf("%w: request model project %q does not match configured project %q", ErrInvalidConfig, resProject, m.project)
		}
		if resLocation != m.location {
			return "", fmt.Errorf("%w: request model location %q does not match configured location %q", ErrInvalidConfig, resLocation, m.location)
		}
	}
	return modelID, nil
}

// GenerateContent implements [model.LLM].
func (m *anthropicModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}
		if req == nil {
			yield(nil, fmt.Errorf("%w: request must not be nil", ErrNoContent))
			return
		}

		params, reqOpts, err := m.buildMessageParams(ctx, req)
		if err != nil {
			yield(nil, err)
			return
		}

		if err := ctx.Err(); err != nil {
			yield(nil, err)
			return
		}

		if stream {
			m.generateStreaming(ctx, params, reqOpts, yield)
			return
		}

		msg, err := m.client.Messages.New(ctx, params, slices.Clone(reqOpts)...)
		if err != nil {
			yield(nil, classifyError(err))
			return
		}
		resp, err := convertMessageToLLMResponse(msg)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(resp, nil)
	}
}
