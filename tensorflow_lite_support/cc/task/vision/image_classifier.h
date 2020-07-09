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

#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow_lite_support/cc/port/integral_types.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/external_file_handler.h"
#include "tensorflow_lite_support/cc/task/vision/core/base_vision_task_api.h"
#include "tensorflow_lite_support/cc/task/vision/core/classification_head.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/proto/bounding_box_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_classifier_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/score_calibration.h"

namespace tflite {
namespace support {
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
// At least one output tensor with `N `classes and either 2 or 4 dimensions:
//   (kTfLiteUInt8/kTfLiteFloat32)
//    - `[1 x N]`
//    - `[1 x 1 x 1 x N]`
//
// An example of such model can be found at:
// https://tfhub.dev/bohemian-visual-recognition-alliance/lite-model/models/mushroom-identification_v1/1
class ImageClassifier : public BaseVisionTaskApi<ClassificationResult> {
 public:
  using BaseVisionTaskApi::BaseVisionTaskApi;

  // Creates an ImageClassifier from the provided options. A non-default
  // OpResolver can be specified in order to support custom Ops or specify a
  // subset of built-in Ops.
  static StatusOr<std::unique_ptr<ImageClassifier>> CreateFromOptions(
      const ImageClassifierOptions& options,
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());

  // Performs actual classification on the provided FrameBuffer.
  StatusOr<ClassificationResult> Classify(const FrameBuffer& frame_buffer);

  // Same as above, except that the classification is performed based on the
  // input region of interest. Note: the region of interest is not clamped, so
  // this method will fail if the region is out of bounds.
  StatusOr<ClassificationResult> Classify(const FrameBuffer& frame_buffer,
                                          const BoundingBox& roi);

 protected:
  // The options used to build this ImageClassifier.
  std::unique_ptr<ImageClassifierOptions> options_;

  // The list of classification heads associated with the corresponding output
  // tensors. Built from TFLite Model Metadata.
  std::vector<ClassificationHead> classification_heads_;

  // Post-processing to transform the raw model outputs into classification
  // results.
  StatusOr<ClassificationResult> Postprocess(
      const std::vector<const TfLiteTensor*>& output_tensors,
      const FrameBuffer& frame_buffer, const BoundingBox& roi) override;

  // Performs sanity checks on the provided ImageClassifierOptions.
  static absl::Status SanityCheckOptions(const ImageClassifierOptions& options);

  // Initializes the ImageClassifier from the provided ImageClassifierOptions,
  // whose ownership is transferred to this object.
  absl::Status Init(std::unique_ptr<ImageClassifierOptions> options);

  // Performs pre-initialization actions.
  virtual absl::Status PreInit();
  // Performs post-initialization actions.
  virtual absl::Status PostInit();

 private:
  // Performs sanity checks on the model outputs and extracts their metadata.
  absl::Status CheckAndSetOutputs();

  // Performs sanity checks on the class whitelist/blacklist and forms the class
  // name set.
  absl::Status CheckAndSetClassNameSet();

  // Initializes the score calibration parameters based on corresponding TFLite
  // Model Metadata, if any.
  absl::Status InitScoreCalibrations();

  // Given a ClassificationResult object containing class indices, fills the
  // name and display name from the label map(s).
  absl::Status FillResultsFromLabelMaps(ClassificationResult* result);

  // The number of output tensors. This corresponds to the number of
  // classification heads.
  int num_outputs_;
  // Whether the model features quantized inference type (QUANTIZED_UINT8). This
  // is currently detected by checking if all output tensors data type is uint8.
  bool has_uint8_outputs_;

  // Set of whitelisted or blacklisted class names.
  struct ClassNameSet {
    absl::flat_hash_set<std::string> values;
    bool is_whitelist;
  };

  // Whitelisted or blacklisted class names based on provided options at
  // construction time. These are used to filter out results during
  // post-processing.
  ClassNameSet class_name_set_;

  // List of score calibration parameters, if any. Built from TFLite Model
  // Metadata.
  std::vector<std::unique_ptr<ScoreCalibration>> score_calibrations_;
};

}  // namespace vision
}  // namespace task
}  // namespace support
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_IMAGE_CLASSIFIER_H_
