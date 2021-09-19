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

#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_LANDMARK_DETECTOR_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_LANDMARK_DETECTOR_H_

#include <memory>
#include <vector>

#include "external/com_google_absl/absl/status/status.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/vision/core/base_vision_task_api.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/proto/landmark_detector_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/landmarks_proto_inc.h"

namespace tflite {
namespace task {
namespace vision {

// Performs landmark detection on images.
//
// The API expects a TFLite model with optional TFLite Model Metadata.
//
// Input tensor:
//  (kTfLiteUInt8)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - only RGB inputs are supported (`channels` is required to be 3).
// Output tensor:
//  (kTfLiteFloat32)
//    - one output tensor with 4 dimensions `[1 x 1 x num_keypoints x 3]`, the
//      last dimension representing keypoint coordinates with predicted
//      confidence score in the form [y, x, score].
//
// The MoveNet model can be found at:
// https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4

class LandmarkDetector : public BaseVisionTaskApi<LandmarkResult> {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  static tflite::support::StatusOr<std::unique_ptr<LandmarkDetector>>
  CreateFromOptions(const LandmarkDetectorOptions& options);

  // Performs actual detection on the provided FrameBuffer.
  //
  // The FrameBuffer can be of any size and any of the supported formats, i.e.
  // RGBA, RGB, NV12, NV21, YV12, YV21. It is automatically pre-processed before
  // inference in order to (and in this order):
  // - resize it (with bilinear interpolation, aspect-ratio *not* preserved) to
  //   the dimensions of the model input tensor,
  // - convert it to the colorspace of the input tensor (i.e. RGB, which is the
  //   only supported colorspace for now),
  // - rotate it according to its `Orientation` so that inference is performed
  //   on an "upright" image.

  tflite::support::StatusOr<LandmarkResult> Detect(
      const FrameBuffer& frame_buffer);

  // Same as above, except that the detection is performed based on the
  // input region of interest. Cropping according to this region of interest is
  // prepended to the pre-processing operations.
  //
  // IMPORTANT: as a consequence of cropping occurring first, the provided
  // region of interest is expressed in the unrotated frame of reference
  // coordinates system, i.e. in `[0, frame_buffer.width) x [0,
  // frame_buffer.height)`, which are the dimensions of the underlying
  // `frame_buffer` data before any `Orientation` flag gets applied. Also, the
  // region of interest is not clamped, so this method will return a non-ok
  // status if the region is out of these bounds.

  tflite::support::StatusOr<LandmarkResult> Detect(
      const FrameBuffer& frame_buffer, const BoundingBox& roi);

 protected:
  // The options used to build this LandmarkDetector.
  std::unique_ptr<LandmarkDetectorOptions> options_;

  // Post-processing to transform the raw model outputs into landmarks
  // results.
  tflite::support::StatusOr<LandmarkResult> Postprocess(
      const std::vector<const TfLiteTensor*>& output_tensors,
      const FrameBuffer& frame_buffer, const BoundingBox& roi) override;

  // Performs sanity checks on the provided LandmarkDetectorOptions.
  static absl::Status SanityCheckOptions(
      const LandmarkDetectorOptions& options);

  // Initializes the LandmarkDetector from the provided LandmarkDetectorOptions,
  // whose ownership is transferred to this object.
  absl::Status Init(std::unique_ptr<LandmarkDetectorOptions> options);

  // Performs pre-initialization actions.
  virtual absl::Status PreInit();

  // Performs Sanity check for output_tensors.
  absl::Status SanityCheckOutputTensors();
};

}  // namespace vision
}  // namespace task
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_LANDMARK_DETECTOR_H_
