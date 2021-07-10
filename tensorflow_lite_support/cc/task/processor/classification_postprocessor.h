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

#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_CLASSIFICATION_POSTPROCESSOR_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_CLASSIFICATION_POSTPROCESSOR_H_

#include <initializer_list>

#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/core/classification_head.h"
#include "tensorflow_lite_support/cc/task/core/score_calibration.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"
#include "tensorflow_lite_support/cc/task/processor/processor.h"
#include "tensorflow_lite_support/cc/task/processor/proto/classification_options.proto.h"
#include "tensorflow_lite_support/cc/task/processor/proto/classifications.proto.h"

namespace tflite {
namespace task {
namespace processor {

// This Postprocessor expects one output tensor with:
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
class ClassificationPostprocessor : public Postprocessor {
 public:
  static absl::StatusOr<std::unique_ptr<ClassificationPostprocessor>> Create(
      core::TfLiteEngine* engine,
      const std::initializer_list<int> output_indices,
      std::unique_ptr<ClassificationOptions> options) {
    auto processer =
        absl::WrapUnique(new ClassificationPostprocessor(std::move(options)));

    static constexpr int tensor_count = 1;
    RETURN_IF_ERROR(
        processer->VerifyAndInit(tensor_count, engine, output_indices));
    RETURN_IF_ERROR(processer->Init());
    return processer;
  }

  absl::Status Postprocess(Classifications* classifications);

 protected:
  ClassificationPostprocessor(std::unique_ptr<ClassificationOptions> options)
      : options_(std::move(options)) {}

  absl::Status Init();

 private:
  // Given a ClassificationResult object containing class indices, fills the
  // name and display name from the label map(s).
  absl::Status FillResultsFromLabelMaps(Classifications* classifications);

  // Released after `Init`.
  std::unique_ptr<ClassificationOptions> options_;

  // The list of classification heads associated with the corresponding output
  // tensors. Built from TFLite Model Metadata.
  ::tflite::task::core::ClassificationHead classification_head_{};

  // Set of allowlisted or denylisted class names.
  struct ClassNameSet {
    absl::flat_hash_set<std::string> values;
    bool is_allowlist;
  };

  // Allowlisted or denylisted class names based on provided options at
  // construction time. These are used to filter out results during
  // post-processing.
  ClassNameSet class_name_set_;

  // Score calibration parameters, if any. Built from TFLite Model
  // Metadata.
  std::unique_ptr<core::ScoreCalibration> score_calibration_;
};

}  // namespace processor
}  // namespace task
}  // namespace tflite
#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_CLASSIFICATION_POSTPROCESSOR_H_
