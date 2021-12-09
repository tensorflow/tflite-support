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

// Process the associated output image tensor and convert it to a FrameBuffer.
// Requirement for the output tensor:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - only RGB inputs are supported (`channels` is required to be 3).
//    - if type is kTfLiteFloat32, NormalizationOptions are required to be
//      attached to the metadata for output de-normalization. Uses input metadata
//      as fallback in case output metadata isn't provided.
class ImagePostprocessor : public Postprocessor {
 public:
  static tflite::support::StatusOr<std::unique_ptr<ImagePostprocessor>>
  Create(core::TfLiteEngine* engine,
         const int output_index,
         const int input_index = -1);

  // Processes the output tensor to an RGB of FrameBuffer type.
  // If output tensor is of type kTfLiteFloat32, denormalize it into [0 - 255]
  // via normalization parameters.
  absl::StatusOr<vision::FrameBuffer> Postprocess();

 private:
  using Postprocessor::Postprocessor;

  // Whether the model features quantized inference type (QUANTIZED_UINT8). This
  // is currently detected by checking if the output tensor data type is uint8.
  bool has_uint8_output_;

  std::unique_ptr<vision::NormalizationOptions> options_;

  absl::Status Init(const int input_index);

  const vision::NormalizationOptions& GetNormalizationOptions() {
    return *options_.get();
  }
};
}  // namespace processor
}  // namespace task
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_IMAGE_POSTPROCESSOR_H_
