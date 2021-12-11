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

  RETURN_IF_ERROR(processor->Init(input_index));
  return processor;
}

absl::Status ImagePostprocessor::Init(const int input_index) {
  if (input_index == -1) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Input image tensor not set. Input index found: %d",
                        input_index),
        tflite::support::TfLiteSupportStatus::kInputTensorNotFoundError);
  }
  ASSIGN_OR_RETURN(
      auto output_specs,
      vision::BuildImageTensorSpecs(*engine_->interpreter(),
                                    *engine_->metadata_extractor(), false));
  options_ = std::make_unique<vision::NormalizationOptions>(
      output_specs.normalization_options.value());
  return absl::OkStatus();
}

absl::StatusOr<vision::FrameBuffer> ImagePostprocessor::Postprocess() {
  has_uint8_output_ = GetTensor()->type == kTfLiteUInt8;
  vision::FrameBuffer::Dimension to_buffer_dimension = {
      GetTensor()->dims->data[2], GetTensor()->dims->data[1]};
  size_t output_byte_size =
      GetBufferByteSize(to_buffer_dimension, vision::FrameBuffer::Format::kRGB);
  std::vector<uint8> postprocessed_data(output_byte_size / sizeof(uint8), 0);

  if (has_uint8_output_) {  // No denormalization required.
    if (GetTensor()->bytes != output_byte_size) {
      return tflite::support::CreateStatusWithPayload(
          absl::StatusCode::kInternal,
          "Size mismatch or unsupported padding bytes between pixel data "
          "and output tensor.");
    }
    const uint8* output_data =
        core::AssertAndReturnTypedTensor<uint8>(GetTensor()).value();
    postprocessed_data.insert(postprocessed_data.begin(), &output_data[0],
                              &output_data[output_byte_size / sizeof(uint8)]);
  } else {  // Denormalize to [0, 255] range.
    if (GetTensor()->bytes / sizeof(float) !=
        output_byte_size / sizeof(uint8)) {
      return tflite::support::CreateStatusWithPayload(
          absl::StatusCode::kInternal,
          "Size mismatch or unsupported padding bytes between pixel data "
          "and output tensor.");
    }

    uint8* denormalized_output_data = postprocessed_data.data();
    const float* output_data =
        core::AssertAndReturnTypedTensor<float>(GetTensor()).value();
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
      /*stride=*/{GetTensor()->dims->data[2] * kRgbPixelBytes, kRgbPixelBytes}};
  auto postprocessed_frame_buffer =
      vision::FrameBuffer::Create({postprocessed_plane}, to_buffer_dimension,
                                  vision::FrameBuffer::Format::kRGB,
                                  vision::FrameBuffer::Orientation::kTopLeft);
  return *postprocessed_frame_buffer.get();
}

}  // namespace processor
}  // namespace task
}  // namespace tflite
