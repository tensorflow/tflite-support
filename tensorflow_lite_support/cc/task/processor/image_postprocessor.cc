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

namespace {

using ::absl::StatusCode;
using ::tflite::metadata::ModelMetadataExtractor;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;

constexpr int kRgbPixelBytes = 3;

}  // namespace

/* static */
tflite::support::StatusOr<std::unique_ptr<ImagePostprocessor>>
ImagePostprocessor::Create(core::TfLiteEngine* engine, const int output_index,
                           const int input_index) {
  ASSIGN_OR_RETURN(auto processor,
                   Processor::Create<ImagePostprocessor>(
                       /* num_expected_tensors = */ 1, engine, {output_index},
                       /* requires_metadata = */ false));

  RETURN_IF_ERROR(processor->Init(input_index, output_index));
  return processor;
}

absl::Status ImagePostprocessor::Init(const int input_index,
                                      const int output_index) {
  if (input_index == -1) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Input image tensor not set. Input index found: %d",
                        input_index),
        tflite::support::TfLiteSupportStatus::kInputTensorNotFoundError);
  }
  const TensorMetadata* metadata = GetTensorMetadata(output_index);
  // Fallback to input metadata if output meta doesn't have norm params.
  ASSIGN_OR_RETURN(
      const tflite::ProcessUnit* normalization_process_unit,
      ModelMetadataExtractor::FindFirstProcessUnit(
          *metadata, tflite::ProcessUnitOptions_NormalizationOptions));
  if (normalization_process_unit == nullptr) {
    metadata =
        engine_->metadata_extractor()->GetInputTensorMetadata(input_index);
  }
  if (!GetTensor(output_index)->data.raw) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInternal,
        absl::StrFormat("Output tensor (%s) has no raw data.",
                        GetTensor(output_index)->name));
  }
  output_tensor_ = GetTensor(output_index);
  ASSIGN_OR_RETURN(auto output_specs,
                   vision::BuildImageTensorSpecs(*engine_->metadata_extractor(),
                                                 metadata, output_tensor_));
  options_ = std::make_unique<vision::NormalizationOptions>(
      output_specs.normalization_options.value());
  return absl::OkStatus();
}

absl::StatusOr<vision::FrameBuffer> ImagePostprocessor::Postprocess() {
  vision::FrameBuffer::Dimension to_buffer_dimension = {
      output_tensor_->dims->data[2], output_tensor_->dims->data[1]};
  size_t output_byte_size =
      GetBufferByteSize(to_buffer_dimension, vision::FrameBuffer::Format::kRGB);
  std::vector<uint8> postprocessed_data(output_byte_size / sizeof(uint8), 0);

  if (output_tensor_->type == kTfLiteUInt8) {  // No denormalization required.
    core::PopulateVector(output_tensor_, &postprocessed_data);
  } else if (output_tensor_->type ==
             kTfLiteFloat32) {  // Denormalize to [0, 255] range.
    uint8* denormalized_output_data = postprocessed_data.data();
    const float* output_data =
        core::AssertAndReturnTypedTensor<float>(output_tensor_).value();
    const auto norm_options = GetNormalizationOptions();

    if (norm_options.num_values == 1) {
      float mean_value = norm_options.mean_values[0];
      float std_value = norm_options.std_values[0];

      for (size_t i = 0; i < output_byte_size / sizeof(uint8);
           ++i, ++denormalized_output_data, ++output_data) {
        *denormalized_output_data = static_cast<uint8>(std::round(std::min(
            255.f, std::max(0.f, (*output_data) * std_value + mean_value))));
      }
    } else {
      for (size_t i = 0; i < output_byte_size / sizeof(uint8);
           ++i, ++denormalized_output_data, ++output_data) {
        *denormalized_output_data = static_cast<uint8>(std::round(std::min(
            255.f,
            std::max(0.f, (*output_data) * norm_options.std_values[i % 3] +
                              norm_options.mean_values[i % 3]))));
      }
    }
  }

  vision::FrameBuffer::Plane postprocessed_plane = {
      /*buffer=*/postprocessed_data.data(),
      /*stride=*/{output_tensor_->dims->data[2] * kRgbPixelBytes,
                  kRgbPixelBytes}};
  auto postprocessed_frame_buffer =
      vision::FrameBuffer::Create({postprocessed_plane}, to_buffer_dimension,
                                  vision::FrameBuffer::Format::kRGB,
                                  vision::FrameBuffer::Orientation::kTopLeft);
  return *postprocessed_frame_buffer.get();
}

}  // namespace processor
}  // namespace task
}  // namespace tflite
