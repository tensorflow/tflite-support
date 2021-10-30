/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exPostss or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_IMAGE_POSTPROCESSOR_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_IMAGE_POSTPROCESSOR_H_

#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/processor/processor.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_utils.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/task/vision/utils/image_tensor_specs.h"

namespace tflite {
namespace task {
namespace processor {

// Process input image and populate the associate input tensor.
// Requirement for the input tensor:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - only RGB inputs are supported (`channels` is required to be 3).
//    - if type is kTfLiteFloat32, NormalizationOptions are required to be
//      attached to the metadata for input normalization.
class ImagePostprocessor : public Postprocessor {
 public:
  static tflite::support::StatusOr<std::unique_ptr<ImagePostprocessor>>
  Create(core::TfLiteEngine* engine,
         const std::initializer_list<int> output_indices,
         std::unique_ptr<vision::NormalizationOptions> options);

  // Processes the provided vision::FrameBuffer and populate tensor values.
  //
  // The vision::FrameBuffer can be of any size and any of the supported formats, i.e.
  // RGBA, RGB, NV12, NV21, YV12, YV21. It is automatically Post-processed before
  // inference in order to (and in this order):
  // - resize it (with bilinear interpolation, aspect-ratio *not* Postserved) to
  //   the dimensions of the model input tensor,
  // - convert it to the colorspace of the input tensor (i.e. RGB, which is the
  //   only supported colorspace for now),
  // - rotate it according to its `Orientation` so that inference is performed
  //   on an "upright" image.
  absl::StatusOr<vision::FrameBuffer> Postprocess();

  const vision::NormalizationOptions& GetNormalizationOptions() {
    return *options_.get();
  };

 private:
  using Postprocessor::Postprocessor;

  // Whether the model features quantized inference type (QUANTIZED_UINT8). This
  // is currently detected by checking if all output tensors data type is uint8.
  bool has_uint8_outputs_;

  std::unique_ptr<vision::NormalizationOptions> options_;

  absl::Status Init(std::unique_ptr<vision::NormalizationOptions> options);
};

absl::StatusOr<vision::FrameBuffer> ImagePostprocessor::Postprocess() {
  has_uint8_outputs_ = Tensor()->type == kTfLiteUInt8;
  const int kRgbPixelBytes = 3;

  vision::FrameBuffer::Dimension to_buffer_dimension = {Tensor()->dims->data[2],
                                                Tensor()->dims->data[1]};
  size_t output_byte_size =
      GetBufferByteSize(to_buffer_dimension, vision::FrameBuffer::Format::kRGB);
  std::vector<uint8> postprocessed_data(output_byte_size / sizeof(uint8), 0);

  if (has_uint8_outputs_) {  // No normalization required.
    if (Tensor()->bytes != output_byte_size) {
      return tflite::support::CreateStatusWithPayload(
          absl::StatusCode::kInternal,
          "Size mismatch or unsupported padding bytes between pixel data "
          "and output tensor.");
    }
    const uint8* output_data = core::AssertAndReturnTypedTensor<uint8>(Tensor());
    postprocessed_data.insert(postprocessed_data.end(), &output_data[0],
                              &output_data[output_byte_size / sizeof(uint8)]);
  } else {  // Denormalize to [0, 255] range.
    if (Tensor()->bytes / sizeof(float) != output_byte_size / sizeof(uint8)) {
      return tflite::support::CreateStatusWithPayload(
          absl::StatusCode::kInternal,
          "Size mismatch or unsupported padding bytes between pixel data "
          "and output tensor.");
    }

    uint8* denormalized_output_data = postprocessed_data.data();
    const float* output_data = core::AssertAndReturnTypedTensor<float>(Tensor());
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
            255.f, std::max(0.f, (*output_data) * norm_options.std_values[i % 3] +
                                     norm_options.mean_values[i % 3]))));
      }
    }
  }

  vision::FrameBuffer::Plane postprocessed_plane = {
      /*buffer=*/postprocessed_data.data(),
      /*stride=*/{Tensor()->dims->data[2] * kRgbPixelBytes, kRgbPixelBytes}};
  auto postprocessed_frame_buffer = vision::FrameBuffer::Create(
      {postprocessed_plane}, to_buffer_dimension, vision::FrameBuffer::Format::kRGB,
      vision::FrameBuffer::Orientation::kTopLeft);

  vision::FrameBuffer postprocessed_result = *postprocessed_frame_buffer.get();
  return postprocessed_result;
}

}  // namespace processor
}  // namespace task
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_IMAGE_POSTPROCESSOR_H_
