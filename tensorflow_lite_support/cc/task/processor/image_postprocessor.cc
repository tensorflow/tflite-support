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

#include "tensorflow_lite_support/cc/task/processor/image_postprocessor.h"

namespace tflite {
namespace task {
namespace processor {

/* static */
tflite::support::StatusOr<std::unique_ptr<ImagePostprocessor>>
ImagePostprocessor::Create(
    core::TfLiteEngine* engine, const std::initializer_list<int> output_indices,
    std::unique_ptr<vision::NormalizationOptions> options) {
  RETURN_IF_ERROR(Postprocessor::SanityCheck(/* num_expected_tensors = */ 1,
                                             engine, output_indices));
  auto processor =
      absl::WrapUnique(new ImagePostprocessor(engine, output_indices));

  RETURN_IF_ERROR(processor->Init(std::move(options)));
  return processor;
}

absl::Status ImagePostprocessor::Init(
    std::unique_ptr<vision::NormalizationOptions> options) {
  options_ = std::move(options);

  int output_index = output_indices_.at(0);
  auto* output_tensor = Tensor();

  // Check tensor dimensions.
  if (output_tensor->dims->size != 4) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Output tensor is expected to have 4 dimensions, found %d.",
            output_tensor->dims->size),
        tflite::support::TfLiteSupportStatus::
            kInvalidOutputTensorDimensionsError);
  }

  if (output_tensor->dims->data[0] != 1) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Expected batch size of 1, found %d.",
                        output_tensor->dims->data[0]),
        tflite::support::TfLiteSupportStatus::
            kInvalidOutputTensorDimensionsError);
  }

  // RGB check.
  if (output_tensor->dims->data[3] != 3) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Expected depth size of 3, found %d.",
                        output_tensor->dims->data[3]),
        tflite::support::TfLiteSupportStatus::
            kInvalidOutputTensorDimensionsError);
  }

  if (output_tensor->type != kTfLiteUInt8 &&
      output_tensor->type != kTfLiteFloat32) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Type mismatch for output tensor %s. Requested one "
                        "of these types: "
                        "kTfLiteUint8/kTfLiteFloat32, got %s.",
                        output_tensor->name,
                        TfLiteTypeGetName(output_tensor->type)),
        tflite::support::TfLiteSupportStatus::kInvalidOutputTensorTypeError);
  }
  return absl::OkStatus();
}

}  // namespace processor
}  // namespace task
}  // namespace tflite
