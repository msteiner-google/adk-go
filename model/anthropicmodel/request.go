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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"maps"
	"net/url"
	"slices"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"google.golang.org/genai"

	"google.golang.org/adk/v2/model"
)

func (m *anthropicModel) buildMessageParams(
	ctx context.Context,
	req *model.LLMRequest,
) (anthropic.MessageNewParams, []option.RequestOption, error) {
	modelID, err := m.resolveModelID(req.Model)
	if err != nil {
		return anthropic.MessageNewParams{}, nil, err
	}

	if err := validateGenerateContentConfig(req.Config, req.Tools); err != nil {
		return anthropic.MessageNewParams{}, nil, err
	}

	maxTokens := m.maxTokens
	if req.Config != nil && req.Config.MaxOutputTokens > 0 {
		maxTokens = int(req.Config.MaxOutputTokens)
	}

	systemBlocks := buildSystemBlocks(req.Config)

	tools, toolChoice, err := buildToolsAndChoice(req.Config, req.Tools)
	if err != nil {
		return anthropic.MessageNewParams{}, nil, err
	}

	messages, err := m.buildMessages(ctx, modelID, req.Contents)
	if err != nil {
		return anthropic.MessageNewParams{}, nil, err
	}

	systemBlocks = applyCacheBreakpoints(m.promptCache, systemBlocks, messages, tools)

	params := anthropic.MessageNewParams{
		Model:        anthropic.Model(modelID),
		MaxTokens:    int64(maxTokens),
		Messages:     messages,
		System:       systemBlocks,
		Tools:        tools,
		ToolChoice:   toolChoice,
		Thinking:     buildThinkingParam(req.Config),
		OutputConfig: buildOutputConfigParam(m.effort),
	}

	if req.Config != nil {
		if req.Config.Temperature != nil {
			params.Temperature = anthropic.Float(float64(*req.Config.Temperature))
		}
		if req.Config.TopP != nil {
			params.TopP = anthropic.Float(float64(*req.Config.TopP))
		}
		if req.Config.TopK != nil {
			params.TopK = anthropic.Int(int64(*req.Config.TopK))
		}
		if len(req.Config.StopSequences) > 0 {
			params.StopSequences = slices.Clone(req.Config.StopSequences)
		}
	}

	var reqOpts []option.RequestOption
	if req.Config != nil && req.Config.HTTPOptions != nil &&
		req.Config.HTTPOptions.Timeout != nil && *req.Config.HTTPOptions.Timeout > 0 {
		reqOpts = append(reqOpts, option.WithRequestTimeout(*req.Config.HTTPOptions.Timeout))
	}

	return params, reqOpts, nil
}

func buildSystemBlocks(cfg *genai.GenerateContentConfig) []anthropic.TextBlockParam {
	if cfg == nil || cfg.SystemInstruction == nil {
		return nil
	}
	var texts []string
	for _, part := range cfg.SystemInstruction.Parts {
		if part != nil && part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	if len(texts) == 0 {
		return nil
	}
	return []anthropic.TextBlockParam{
		{Text: strings.Join(texts, "\n\n")},
	}
}

func toClaudeRole(role string) anthropic.MessageParamRole {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case genai.RoleModel, "assistant":
		return anthropic.MessageParamRoleAssistant
	default:
		return anthropic.MessageParamRoleUser
	}
}

func (m *anthropicModel) buildMessages(
	ctx context.Context,
	modelID string,
	contents []*genai.Content,
) ([]anthropic.MessageParam, error) {
	if len(contents) == 0 {
		return nil, fmt.Errorf("%w: contents must not be empty", ErrNoContent)
	}

	// Find the index of the newest (last) entry in contents.
	newestIdx := len(contents) - 1
	if contents[newestIdx] == nil || len(contents[newestIdx].Parts) == 0 {
		return nil, fmt.Errorf("%w: newest message in contents is empty", ErrNoContent)
	}

	sanitizer := newToolUseIDSanitizer()
	var messages []anthropic.MessageParam

	for i, content := range contents {
		if content == nil || len(content.Parts) == 0 {
			continue
		}
		role := toClaudeRole(content.Role)
		blocks, droppedUnsupported, hadEmptyText := m.convertContentParts(ctx, modelID, role, content.Parts, sanitizer)

		if len(blocks) == 0 && hadEmptyText && !droppedUnsupported && role == anthropic.MessageParamRoleUser {
			blocks = append(blocks, anthropic.NewTextBlock("."))
		}

		if len(blocks) == 0 {
			if i == newestIdx {
				if droppedUnsupported {
					return nil, fmt.Errorf("%w: %w: newest message has no sendable content after skipping unsupported parts", ErrNoContent, ErrUnsupportedContent)
				}
				return nil, fmt.Errorf("%w: newest message has no sendable content", ErrNoContent)
			}
			continue
		}

		if role == anthropic.MessageParamRoleAssistant {
			blocks = orderAssistantBlocks(blocks)
		}

		// Coalesce adjacent messages with the same role so conversation turns
		// strictly alternate between user and assistant.
		if len(messages) > 0 && messages[len(messages)-1].Role == role {
			merged := append(messages[len(messages)-1].Content, blocks...)
			if role == anthropic.MessageParamRoleAssistant {
				merged = orderAssistantBlocks(merged)
			}
			messages[len(messages)-1].Content = merged
		} else {
			messages = append(messages, anthropic.MessageParam{
				Role:    role,
				Content: blocks,
			})
		}
	}

	if len(messages) == 0 {
		return nil, fmt.Errorf("%w: no sendable messages in request", ErrNoContent)
	}

	return messages, nil
}

// orderAssistantBlocks places thinking and redacted_thinking blocks before
// text and tool_use blocks on assistant turns, as required by Claude.
func orderAssistantBlocks(blocks []anthropic.ContentBlockParamUnion) []anthropic.ContentBlockParamUnion {
	var thinkingBlocks []anthropic.ContentBlockParamUnion
	var otherBlocks []anthropic.ContentBlockParamUnion
	for _, b := range blocks {
		if b.OfThinking != nil || b.OfRedactedThinking != nil {
			thinkingBlocks = append(thinkingBlocks, b)
		} else {
			otherBlocks = append(otherBlocks, b)
		}
	}
	if len(thinkingBlocks) == 0 || len(otherBlocks) == 0 {
		return blocks
	}
	out := make([]anthropic.ContentBlockParamUnion, 0, len(blocks))
	out = append(out, thinkingBlocks...)
	out = append(out, otherBlocks...)
	return out
}

func (m *anthropicModel) convertContentParts(
	ctx context.Context,
	modelID string,
	role anthropic.MessageParamRole,
	parts []*genai.Part,
	sanitizer *toolUseIDSanitizer,
) (blocks []anthropic.ContentBlockParamUnion, droppedUnsupported, hadEmptyText bool) {
	for _, part := range parts {
		if part == nil {
			continue
		}

		// 1. Thought parts (thinking and redacted_thinking).
		if part.Thought {
			if role != anthropic.MessageParamRoleAssistant {
				// Claude only accepts thinking blocks on assistant turns.
				continue
			}
			if part.Text != "" {
				if len(part.ThoughtSignature) == 0 {
					// Strip unsigned thinking blocks when replaying history, as Claude rejects them.
					continue
				}
				blocks = append(blocks, anthropic.NewThinkingBlock(string(part.ThoughtSignature), part.Text))
				continue
			}
			if len(part.ThoughtSignature) > 0 {
				blocks = append(blocks, anthropic.NewRedactedThinkingBlock(string(part.ThoughtSignature)))
				continue
			}
			continue
		}

		// 2. Content-free foreign thought signature (e.g. from a Gemini turn in history).
		if isContentFreeSignature(part) {
			m.warnSkippedPart(ctx, modelID, role, "thought_signature", "", "")
			continue
		}

		// 3. FunctionCall -> tool_use.
		if part.FunctionCall != nil {
			fc := part.FunctionCall
			toolID := sanitizer.sanitize(fc.ID, fc.Name, true)
			inputMap := make(map[string]any, len(fc.Args))
			maps.Copy(inputMap, fc.Args)
			blocks = append(blocks, anthropic.NewToolUseBlock(toolID, inputMap, fc.Name))
			continue
		}

		// 4. FunctionResponse -> tool_result.
		if part.FunctionResponse != nil {
			block := m.convertFunctionResponsePart(ctx, modelID, role, part.FunctionResponse, sanitizer)
			blocks = append(blocks, block)
			continue
		}

		// 5. InlineData (images, PDF, plain text).
		if part.InlineData != nil {
			block, ok := m.convertInlineDataPart(ctx, modelID, role, part.InlineData)
			if !ok {
				droppedUnsupported = true
				continue
			}
			blocks = append(blocks, block)
			continue
		}

		// 6. FileData (https:// URLs for images and PDFs on BackendAnthropic).
		if part.FileData != nil {
			block, ok := m.convertFileDataPart(ctx, modelID, role, part.FileData)
			if !ok {
				droppedUnsupported = true
				continue
			}
			blocks = append(blocks, block)
			continue
		}

		// 7. ExecutableCode and CodeExecutionResult (converted to text blocks like adk-python).
		if part.ExecutableCode != nil {
			blocks = append(blocks, anthropic.NewTextBlock("Code:```python\n"+part.ExecutableCode.Code+"\n```"))
			continue
		}
		if part.CodeExecutionResult != nil {
			blocks = append(blocks, anthropic.NewTextBlock("Execution Result:```code_output\n"+part.CodeExecutionResult.Output+"\n```"))
			continue
		}

		// 8. Unsupported server-side tool / audio / video metadata parts.
		if part.ToolCall != nil {
			m.warnSkippedPart(ctx, modelID, role, "tool_call", "", "")
			droppedUnsupported = true
			continue
		}
		if part.ToolResponse != nil {
			m.warnSkippedPart(ctx, modelID, role, "tool_response", "", "")
			droppedUnsupported = true
			continue
		}
		if part.AudioTranscription != nil {
			m.warnSkippedPart(ctx, modelID, role, "audio_transcription", "", "")
			droppedUnsupported = true
			continue
		}
		if part.VideoMetadata != nil && part.Text == "" {
			m.warnSkippedPart(ctx, modelID, role, "video_metadata", "", "")
			droppedUnsupported = true
			continue
		}

		// 9. Plain text.
		if part.Text != "" {
			blocks = append(blocks, anthropic.NewTextBlock(part.Text))
			continue
		}

		hadEmptyText = true
	}

	return blocks, droppedUnsupported, hadEmptyText
}

func isContentFreeSignature(part *genai.Part) bool {
	if len(part.ThoughtSignature) == 0 || part.Thought || part.Text != "" {
		return false
	}
	return part.FunctionCall == nil &&
		part.FunctionResponse == nil &&
		part.InlineData == nil &&
		part.FileData == nil &&
		part.ExecutableCode == nil &&
		part.CodeExecutionResult == nil &&
		part.ToolCall == nil &&
		part.ToolResponse == nil &&
		part.AudioTranscription == nil
}

func (m *anthropicModel) convertInlineDataPart(
	ctx context.Context,
	modelID string,
	role anthropic.MessageParamRole,
	blob *genai.Blob,
) (anthropic.ContentBlockParamUnion, bool) {
	mimeType := normalizeMIMEType(blob.MIMEType)
	if role != anthropic.MessageParamRoleUser {
		m.warnSkippedPart(ctx, modelID, role, "inline_data", mimeType, "")
		return anthropic.ContentBlockParamUnion{}, false
	}
	if len(blob.Data) == 0 {
		m.warnSkippedPart(ctx, modelID, role, "inline_data", mimeType, "")
		return anthropic.ContentBlockParamUnion{}, false
	}

	switch {
	case isSupportedImageMIME(mimeType):
		encoded := base64.StdEncoding.EncodeToString(blob.Data)
		return anthropic.NewImageBlockBase64(mimeType, encoded), true
	case mimeType == "application/pdf":
		encoded := base64.StdEncoding.EncodeToString(blob.Data)
		return anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
			Data: encoded,
		}), true
	case mimeType == "text/plain":
		return anthropic.NewDocumentBlock(anthropic.PlainTextSourceParam{
			Data: string(blob.Data),
		}), true
	default:
		m.warnSkippedPart(ctx, modelID, role, "inline_data", mimeType, "")
		return anthropic.ContentBlockParamUnion{}, false
	}
}

func (m *anthropicModel) convertFileDataPart(
	ctx context.Context,
	modelID string,
	role anthropic.MessageParamRole,
	fd *genai.FileData,
) (anthropic.ContentBlockParamUnion, bool) {
	mimeType := normalizeMIMEType(fd.MIMEType)
	rawURI := strings.TrimSpace(fd.FileURI)
	scheme := extractURIScheme(rawURI)

	if role != anthropic.MessageParamRoleUser || m.backend != BackendAnthropic || scheme != "https" || !isValidHTTPSURL(rawURI) {
		m.warnSkippedPart(ctx, modelID, role, "file_data", mimeType, scheme)
		return anthropic.ContentBlockParamUnion{}, false
	}

	switch {
	case isSupportedImageMIME(mimeType):
		return anthropic.NewImageBlock(anthropic.URLImageSourceParam{
			URL: rawURI,
		}), true
	case mimeType == "application/pdf":
		return anthropic.NewDocumentBlock(anthropic.URLPDFSourceParam{
			URL: rawURI,
		}), true
	default:
		m.warnSkippedPart(ctx, modelID, role, "file_data", mimeType, scheme)
		return anthropic.ContentBlockParamUnion{}, false
	}
}

func (m *anthropicModel) convertFunctionResponsePart(
	ctx context.Context,
	modelID string,
	role anthropic.MessageParamRole,
	fr *genai.FunctionResponse,
	sanitizer *toolUseIDSanitizer,
) anthropic.ContentBlockParamUnion {
	toolUseID := sanitizer.sanitize(fr.ID, fr.Name, false)
	textContent := serializeFunctionResponseData(fr.Response)
	mediaBlocks := m.convertFunctionResponseMedia(ctx, modelID, role, fr.Parts)

	var resultContent []anthropic.ToolResultBlockParamContentUnion
	if textContent != "" || len(mediaBlocks) == 0 {
		resultContent = append(resultContent, anthropic.ToolResultBlockParamContentUnion{
			OfText: &anthropic.TextBlockParam{Text: textContent},
		})
	}
	resultContent = append(resultContent, mediaBlocks...)

	return anthropic.ContentBlockParamUnion{
		OfToolResult: &anthropic.ToolResultBlockParam{
			ToolUseID: toolUseID,
			Content:   resultContent,
			IsError:   anthropic.Bool(false),
		},
	}
}

func serializeFunctionResponseData(respData map[string]any) string {
	if len(respData) == 0 {
		return ""
	}
	if rawContent, ok := respData["content"]; ok && rawContent != nil {
		switch v := rawContent.(type) {
		case []any:
			if len(v) > 0 {
				items := make([]string, 0, len(v))
				for _, item := range v {
					if itemMap, ok := item.(map[string]any); ok {
						if t, _ := itemMap["type"].(string); t == "text" {
							if txt, ok := itemMap["text"].(string); ok {
								items = append(items, txt)
								continue
							}
						}
					}
					items = append(items, formatArbitraryValue(item))
				}
				return strings.Join(items, "\n")
			}
		case string:
			if v != "" {
				return v
			}
		}
	}

	if len(respData) == 1 {
		if resultVal, ok := respData["result"]; ok && resultVal != nil {
			return formatArbitraryValue(resultVal)
		}
	}

	if encoded, err := json.Marshal(respData); err == nil {
		return string(encoded)
	}
	return ""
}

func formatArbitraryValue(val any) string {
	switch v := val.(type) {
	case string:
		return v
	case map[string]any, []any:
		if encoded, err := json.Marshal(v); err == nil {
			return string(encoded)
		}
		return fmt.Sprint(v)
	default:
		return fmt.Sprint(v)
	}
}

func (m *anthropicModel) convertFunctionResponseMedia(
	ctx context.Context,
	modelID string,
	role anthropic.MessageParamRole,
	parts []*genai.FunctionResponsePart,
) []anthropic.ToolResultBlockParamContentUnion {
	var out []anthropic.ToolResultBlockParamContentUnion
	for _, p := range parts {
		if p == nil {
			continue
		}
		if p.FileData != nil {
			mimeType := normalizeMIMEType(p.FileData.MIMEType)
			rawURI := strings.TrimSpace(p.FileData.FileURI)
			scheme := extractURIScheme(rawURI)
			if role != anthropic.MessageParamRoleUser || m.backend != BackendAnthropic || scheme != "https" || !isValidHTTPSURL(rawURI) {
				m.warnSkippedPart(ctx, modelID, role, "function_response_file_data", mimeType, scheme)
				continue
			}
			switch {
			case isSupportedImageMIME(mimeType):
				out = append(out, anthropic.ToolResultBlockParamContentUnion{
					OfImage: &anthropic.ImageBlockParam{
						Source: anthropic.ImageBlockParamSourceUnion{
							OfURL: &anthropic.URLImageSourceParam{
								URL: rawURI,
							},
						},
					},
				})
			case mimeType == "application/pdf":
				out = append(out, anthropic.ToolResultBlockParamContentUnion{
					OfDocument: &anthropic.DocumentBlockParam{
						Source: anthropic.DocumentBlockParamSourceUnion{
							OfURL: &anthropic.URLPDFSourceParam{
								URL: rawURI,
							},
						},
					},
				})
			default:
				m.warnSkippedPart(ctx, modelID, role, "function_response_file_data", mimeType, scheme)
			}
			continue
		}
		if p.InlineData == nil {
			continue
		}
		mimeType := normalizeMIMEType(p.InlineData.MIMEType)
		if len(p.InlineData.Data) == 0 {
			m.warnSkippedPart(ctx, modelID, role, "function_response_inline_data", mimeType, "")
			continue
		}
		switch {
		case isSupportedImageMIME(mimeType):
			encoded := base64.StdEncoding.EncodeToString(p.InlineData.Data)
			out = append(out, anthropic.ToolResultBlockParamContentUnion{
				OfImage: &anthropic.ImageBlockParam{
					Source: anthropic.ImageBlockParamSourceUnion{
						OfBase64: &anthropic.Base64ImageSourceParam{
							Data:      encoded,
							MediaType: anthropic.Base64ImageSourceMediaType(mimeType),
						},
					},
				},
			})
		case mimeType == "application/pdf":
			encoded := base64.StdEncoding.EncodeToString(p.InlineData.Data)
			out = append(out, anthropic.ToolResultBlockParamContentUnion{
				OfDocument: &anthropic.DocumentBlockParam{
					Source: anthropic.DocumentBlockParamSourceUnion{
						OfBase64: &anthropic.Base64PDFSourceParam{
							Data: encoded,
						},
					},
				},
			})
		case mimeType == "text/plain":
			out = append(out, anthropic.ToolResultBlockParamContentUnion{
				OfDocument: &anthropic.DocumentBlockParam{
					Source: anthropic.DocumentBlockParamSourceUnion{
						OfText: &anthropic.PlainTextSourceParam{
							Data: string(p.InlineData.Data),
						},
					},
				},
			})
		default:
			m.warnSkippedPart(ctx, modelID, role, "function_response_inline_data", mimeType, "")
		}
	}
	return out
}

func (m *anthropicModel) warnSkippedPart(
	ctx context.Context,
	modelID string,
	role anthropic.MessageParamRole,
	partKind string,
	mimeType string,
	uriScheme string,
) {
	m.logger().WarnContext(ctx, "skipping unsupported content part for Claude request",
		"backend", string(m.backend),
		"model", modelID,
		"role", string(role),
		"part_kind", partKind,
		"mime_type", mimeType,
		"uri_scheme", uriScheme,
	)
}

func normalizeMIMEType(raw string) string {
	base, _, _ := strings.Cut(raw, ";")
	return strings.ToLower(strings.TrimSpace(base))
}

func isSupportedImageMIME(mimeType string) bool {
	switch mimeType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}

func extractURIScheme(rawURI string) string {
	trimmed := strings.TrimSpace(rawURI)
	if trimmed == "" {
		return ""
	}
	if idx := strings.Index(trimmed, "://"); idx > 0 {
		return strings.ToLower(trimmed[:idx])
	}
	if parsed, err := url.Parse(trimmed); err == nil && parsed.Scheme != "" {
		return strings.ToLower(parsed.Scheme)
	}
	return "unknown"
}

func isValidHTTPSURL(rawURI string) bool {
	parsed, err := url.Parse(rawURI)
	return err == nil && strings.EqualFold(parsed.Scheme, "https") && parsed.Host != ""
}
