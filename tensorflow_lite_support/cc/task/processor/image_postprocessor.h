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
//      attached to the metadata for input normalization.
class ImagePostprocessor : public Postprocessor {
 public:
  static tflite::support::StatusOr<std::unique_ptr<ImagePostprocessor>>
  Create(core::TfLiteEngine* engine,
         const std::initializer_list<int> output_indices,
         const std::initializer_list<int> input_indices);

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

 private:
  using Postprocessor::Postprocessor;

  // Whether the model features quantized inference type (QUANTIZED_UINT8). This
  // is currently detected by checking if all output tensors data type is uint8.
  bool has_uint8_outputs_;

  std::unique_ptr<vision::NormalizationOptions> options_;

  absl::Status Init(const std::vector<int>& input_indices);

  const vision::NormalizationOptions& GetNormalizationOptions() {
    return *options_.get();
  }
};
}  // namespace processor
}  // namespace task
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_IMAGE_POSTPROCESSOR_H_
