/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_IMAGE_CLASSIFIER_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_IMAGE_CLASSIFIER_H_

#include <initializer_list>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/shims/cc/kernels/register.h"
#include "tensorflow_lite_support/cc/port/integral_types.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/external_file_handler.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"
#include "tensorflow_lite_support/cc/task/processor/classification_postprocessor.h"
#include "tensorflow_lite_support/cc/task/vision/core/base_vision_task_api.h"
#include "tensorflow_lite_support/cc/task/vision/core/classification_head.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/proto/bounding_box_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_classifier_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/score_calibration.h"

namespace tflite {
namespace task {
namespace vision {

// Performs classification on images.
//
// The API expects a TFLite model with optional, but strongly recommended,
// TFLite Model Metadata.
//
// Input tensor:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - image input of size `[batch x height x width x channels]`.
//    - batch inference is not supported (`batch` is required to be 1).
//    - only RGB inputs are supported (`channels` is required to be 3).
//    - if type is kTfLiteFloat32, NormalizationOptions are required to be
//      attached to the metadata for input normalization.
// At least one output tensor with:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    -  `N `classes and either 2 or 4 dimensions, i.e. `[1 x N]` or
//       `[1 x 1 x 1 x N]`
//    - optional (but recommended) label map(s) as AssociatedFile-s with type
//      TENSOR_AXIS_LABELS, containing one label per line. The first such
//      AssociatedFile (if any) is used to fill the `class_name` field of the
//      results. The `display_name` field is filled from the AssociatedFile (if
//      any) whose locale matches the `display_names_locale` field of the
//      `ImageClassifierOptions` used at creation time ("en" by default, i.e.
//      English). If none of these are available, only the `index` field of the
//      results will be filled.
//
// An example of such model can be found at:
// https://tfhub.dev/bohemian-visual-recognition-alliance/lite-model/models/mushroom-identification_v1/1
//
// A CLI demo tool is available for easily trying out this API, and provides
// example usage. See:
// examples/task/vision/desktop/image_classifier_demo.cc
class ImageClassifier : public BaseVisionTaskApi<ClassificationResult> {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an ImageClassifier from the provided options. A non-default
  // OpResolver can be specified in order to support custom Ops or specify a
  // subset of built-in Ops.
  static tflite::support::StatusOr<std::unique_ptr<ImageClassifier>>
  CreateFromOptions(
      const ImageClassifierOptions& options,
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite_shims::ops::builtin::BuiltinOpResolver>());

  // Performs actual classification on the provided FrameBuffer.
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
  tflite::support::StatusOr<ClassificationResult> Classify(
      const FrameBuffer& frame_buffer);

  // Same as above, except that the classification is performed based on the
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
  tflite::support::StatusOr<ClassificationResult> Classify(
      const FrameBuffer& frame_buffer, const BoundingBox& roi);

 protected:
  absl::StatusOr<std::unique_ptr<processor::ClassificationPostprocessor>>
  CreatePostprocessor(core::TfLiteEngine* engine,
                      const std::initializer_list<int> output_indices,
                      const ImageClassifierOptions& options);

  // The options used to build this ImageClassifier.
  std::unique_ptr<ImageClassifierOptions> options_;

  std::vector<std::unique_ptr<processor::ClassificationPostprocessor>>
      postprocessors_;

  // Post-processing to transform the raw model outputs into classification
  // results.
  tflite::support::StatusOr<ClassificationResult> Postprocess(
      const std::vector<const TfLiteTensor*>& output_tensors,
      const FrameBuffer& frame_buffer, const BoundingBox& roi) override;

  // Performs sanity checks on the provided ImageClassifierOptions.
  static absl::Status SanityCheckOptions(const ImageClassifierOptions& options);

  // Initializes the ImageClassifier from the provided ImageClassifierOptions,
  // whose ownership is transferred to this object.
  absl::Status Init(std::unique_ptr<ImageClassifierOptions> options);

  // Performs pre-initialization actions.
  virtual absl::Status PreInit();
};

}  // namespace vision
}  // namespace task
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_IMAGE_CLASSIFIER_H_
