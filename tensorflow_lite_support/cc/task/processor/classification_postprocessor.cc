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

#include "tensorflow_lite_support/cc/task/processor/classification_postprocessor.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/core/label_map_item.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"

namespace tflite {
namespace task {
namespace processor {

namespace {

using ::absl::StatusCode;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::core::BuildClassificationHead;
using ::tflite::task::core::LabelMapItem;
using ::tflite::task::core::ScoreCalibration;

}  // namespace

absl::Status ClassificationPostprocessor::Init() {
  // Sanity check options
  if (options_->max_results() == 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "Invalid `max_results` option: value must be != 0",
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  if (options_->class_name_allowlist_size() > 0 &&
      options_->class_name_denylist_size() > 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "`class_name_allowlist` and `class_name_denylist` are mutually "
        "exclusive options.",
        TfLiteSupportStatus::kInvalidArgumentError);
  }

  ASSIGN_OR_RETURN(
      classification_head_,
      BuildClassificationHead(*engine_->metadata_extractor(), *Metadata(),
                              options_->display_names_locale()));

  // Sanity check output tensors
  const TfLiteTensor* output_tensor = Tensor();
  const int num_dimensions = output_tensor->dims->size;
  if (num_dimensions == 4) {
    if (output_tensor->dims->data[1] != 1 ||
        output_tensor->dims->data[2] != 1) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Unexpected WxH sizes for output index %d: got "
                          "%dx%d, expected 1x1.",
                          output_tensor_indices_.at(0),
                          output_tensor->dims->data[2],
                          output_tensor->dims->data[1]),
          TfLiteSupportStatus::kInvalidOutputTensorDimensionsError);
    }
  } else if (num_dimensions != 2) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Unexpected number of dimensions for output index %d: got %dD, "
            "expected either 2D (BxN with B=1) or 4D (BxHxWxN with B=1, W=1, "
            "H=1).",
            output_tensor_indices_.at(0), num_dimensions),
        TfLiteSupportStatus::kInvalidOutputTensorDimensionsError);
  }
  if (output_tensor->dims->data[0] != 1) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("The output array is expected to have a batch size "
                        "of 1. Got %d for output index %d.",
                        output_tensor->dims->data[0],
                        output_tensor_indices_.at(0)),
        TfLiteSupportStatus::kInvalidOutputTensorDimensionsError);
  }
  int num_classes = output_tensor->dims->data[num_dimensions - 1];
  // If label map is not set, build a default one based on model
  // introspection. This happens if a model with partial or no metadata was
  // provided through the `model_file_with_metadata` options field.
  if (classification_head_.label_map_items.empty()) {
    classification_head_.label_map_items.reserve(num_classes);
    for (int class_index = 0; class_index < num_classes; ++class_index) {
      classification_head_.label_map_items.emplace_back(LabelMapItem{});
    }
  }
  int num_label_map_items = classification_head_.label_map_items.size();
  if (num_classes != num_label_map_items) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Got %d class(es) for output index %d, expected %d "
                        "according to the label map.",
                        output_tensor->dims->data[num_dimensions - 1],
                        output_tensor_indices_.at(0), num_label_map_items),
        TfLiteSupportStatus::kMetadataInconsistencyError);
  }
  if (output_tensor->type != kTfLiteUInt8 &&
      output_tensor->type != kTfLiteFloat32) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Type mismatch for output tensor %s. Requested one "
                        "of these types: "
                        "kTfLiteUint8/kTfLiteFloat32, got %s.",
                        output_tensor->name,
                        TfLiteTypeGetName(output_tensor->type)),
        TfLiteSupportStatus::kInvalidOutputTensorTypeError);
  }

  // Set class name set
  if (options_->class_name_denylist_size() != 0 ||
      options_->class_name_allowlist_size() != 0) {
    // Before processing class names allowlist or denylist from the input
    // options create a set with _all_ known class names from the label map(s).
    absl::flat_hash_set<std::string> head_class_names;
    for (const auto& item : classification_head_.label_map_items) {
      if (!item.name.empty()) {
        head_class_names.insert(item.name);
      }
    }

    if (head_class_names.empty()) {
      std::string name = classification_head_.name;
      if (name.empty()) {
        name = absl::StrFormat("#%d", output_tensor_indices_.at(0));
      }
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat(
              "Using `class_name_allowlist` or `class_name_denylist` "
              "requires labels to be present but none was found for "
              "classification head: %s",
              name),
          TfLiteSupportStatus::kMetadataMissingLabelsError);
    }

    class_name_set_.is_allowlist = options_->class_name_allowlist_size() > 0;
    const auto& class_names = class_name_set_.is_allowlist
                                  ? options_->class_name_allowlist()
                                  : options_->class_name_denylist();

    // Note: duplicate or unknown classes are just ignored.
    class_name_set_.values.clear();
    for (const auto& class_name : class_names) {
      if (!head_class_names.contains(class_name)) {
        continue;
      }
      class_name_set_.values.insert(class_name);
    }

    if (class_name_set_.values.empty()) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat(
              "Invalid class names specified via `class_name_%s`: none match "
              "with model labels.",
              class_name_set_.is_allowlist ? "allowlsit" : "denylist"),
          TfLiteSupportStatus::kInvalidArgumentError);
    }
  }

  // Set score calibration
  if (classification_head_.calibration_params.has_value()) {
    // Use a specific default score instead of the one specified by default in
    // cc/task/vision/utils/score_calibration.h. See `kDefaultCalibratedScore`
    // documentation for more details.
    classification_head_.calibration_params->default_score =
        kDefaultCalibratedScore;

    score_calibration_ = absl::make_unique<ScoreCalibration>();
    if (score_calibration_ == nullptr) {
      return CreateStatusWithPayload(
          StatusCode::kInternal, "Could not create score calibration object.");
    }

    RETURN_IF_ERROR(score_calibration_->InitializeFromParameters(
        classification_head_.calibration_params.value()));
  }
  return absl::OkStatus();
}


}  // namespace processor
}  // namespace task
}  // namespace tflite
