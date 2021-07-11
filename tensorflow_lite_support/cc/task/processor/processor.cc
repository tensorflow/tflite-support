/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow_lite_support/cc/task/processor/processor.h"

namespace tflite {
namespace task {
namespace processor {

/* static */
absl::Status Preprocessor::SanityCheck(
    int num_expected_tensors, core::TfLiteEngine* engine,
    const std::initializer_list<int> input_indices) {
  if (input_indices.size() != num_expected_tensors) {
    return support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Preprocessor can handle %d tensors, "
                        "got: %d tensors.",
                        num_expected_tensors, input_indices.size()));
  }
  for (auto* p = input_indices.begin(); p < input_indices.end(); p++) {
    int input_index = *p;
    if (input_index < 0 ||
        input_index >= engine->InputCount(engine->interpreter())) {
      return support::CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat(
              "Invalid input_index: %d. Model has %d input tensors.",
              input_index, engine->InputCount(engine->interpreter())));
    }
    auto* metadata =
        engine->metadata_extractor()->GetInputTensorMetadata(input_index);
    if (metadata == nullptr) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Input tensor %d is missing TensorMetadata.",
                          input_index),
          support::TfLiteSupportStatus::kMetadataNotFoundError);
    }
  }

  return absl::OkStatus();
}

/* static */
absl::Status Postprocessor::SanityCheck(
    int num_expected_tensors, core::TfLiteEngine* engine,
    const std::initializer_list<int> output_indices) {
  if (output_indices.size() != num_expected_tensors) {
    return support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Postprocessor can handle %d tensors, "
                        "got: %d tensors.",
                        num_expected_tensors, output_indices.size()));
  }
  for (auto* p = output_indices.begin(); p < output_indices.end(); p++) {
    int output_index = *p;
    if (output_index < 0 ||
        output_index >= engine->OutputCount(engine->interpreter())) {
      return support::CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat(
              "Invalid output_index: %d. Model has %d output tensors.",
              output_index, engine->OutputCount(engine->interpreter())));
    }
    auto* metadata =
        engine->metadata_extractor()->GetOutputTensorMetadata(output_index);
    if (metadata == nullptr) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          absl::StrFormat("Output tensor %d is missing TensorMetadata.",
                          output_index),
          support::TfLiteSupportStatus::kMetadataNotFoundError);
    }
  }

  return absl::OkStatus();
}

}  // namespace processor
}  // namespace task
}  // namespace tflite
