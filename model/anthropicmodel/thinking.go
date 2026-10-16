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
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"google.golang.org/genai"
)

func validateThinkingConfig(tc *genai.ThinkingConfig) error {
	if tc == nil {
		return nil
	}
	if tc.ThinkingLevel != "" {
		return fmt.Errorf("%w: ThinkingConfig.ThinkingLevel is not supported on Claude; configure ClientConfig.Effort instead", ErrInvalidConfig)
	}
	if tc.ThinkingBudget == nil {
		return fmt.Errorf("%w: ThinkingConfig.ThinkingBudget must be set explicitly when ThinkingConfig is provided (0 to disable, >0 for manual budget, <0 for adaptive thinking)", ErrInvalidConfig)
	}
	// IncludeThoughts is ignored by design; thought parts are always returned when thinking is active.
	return nil
}

func isThinkingEnabled(tc *genai.ThinkingConfig) bool {
	return tc != nil && tc.ThinkingBudget != nil && *tc.ThinkingBudget != 0
}

func buildThinkingParam(cfg *genai.GenerateContentConfig) anthropic.ThinkingConfigParamUnion {
	if cfg == nil || cfg.ThinkingConfig == nil || cfg.ThinkingConfig.ThinkingBudget == nil {
		return anthropic.ThinkingConfigParamUnion{}
	}
	budget := *cfg.ThinkingConfig.ThinkingBudget
	switch {
	case budget == 0:
		disabled := anthropic.NewThinkingConfigDisabledParam()
		return anthropic.ThinkingConfigParamUnion{OfDisabled: &disabled}
	case budget < 0:
		return anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &anthropic.ThinkingConfigAdaptiveParam{
				Display: anthropic.ThinkingConfigAdaptiveDisplaySummarized,
			},
		}
	default:
		return anthropic.ThinkingConfigParamOfEnabled(int64(budget))
	}
}

func buildOutputConfigParam(effort Effort) anthropic.OutputConfigParam {
	if effort == "" {
		return anthropic.OutputConfigParam{}
	}
	return anthropic.OutputConfigParam{
		Effort: anthropic.OutputConfigEffort(effort),
	}
}
