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

#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/integral_types.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"
#include "tensorflow_lite_support/cc/task/processor/classification_postprocessor.h"
#include "tensorflow_lite_support/cc/task/processor/proto/classification_options.proto.h"
#include "tensorflow_lite_support/cc/task/vision/core/label_map_item.h"
#include "tensorflow_lite_support/cc/task/vision/proto/class_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_utils.h"
#include "tensorflow_lite_support/metadata/cc/metadata_extractor.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace task {
namespace vision {

namespace {

using ::absl::StatusCode;
using ::tflite::metadata::ModelMetadataExtractor;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::core::AssertAndReturnTypedTensor;
using ::tflite::task::core::TaskAPIFactory;
using ::tflite::task::core::TfLiteEngine;

}  // namespace

/* static */
StatusOr<std::unique_ptr<ImageClassifier>> ImageClassifier::CreateFromOptions(
    const ImageClassifierOptions& options,
    std::unique_ptr<tflite::OpResolver> resolver) {
  RETURN_IF_ERROR(SanityCheckOptions(options));

  // Copy options to ensure the ExternalFile outlives the constructed object.
  auto options_copy = absl::make_unique<ImageClassifierOptions>(options);

  std::unique_ptr<ImageClassifier> image_classifier;
  if (options_copy->has_model_file_with_metadata()) {
    ASSIGN_OR_RETURN(
        image_classifier,
        TaskAPIFactory::CreateFromExternalFileProto<ImageClassifier>(
            &options_copy->model_file_with_metadata(), std::move(resolver),
            options_copy->num_threads(), options_copy->compute_settings()));
  } else if (options_copy->base_options().has_model_file()) {
    ASSIGN_OR_RETURN(image_classifier,
                     TaskAPIFactory::CreateFromBaseOptions<ImageClassifier>(
                         &options_copy->base_options(), std::move(resolver)));
  } else {
    // Should never happen because of SanityCheckOptions.
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 0."),
        TfLiteSupportStatus::kInvalidArgumentError);
  }

  RETURN_IF_ERROR(image_classifier->Init(std::move(options_copy)));

  return image_classifier;
}

/* static */
absl::Status ImageClassifier::SanityCheckOptions(
    const ImageClassifierOptions& options) {
  int num_input_models = (options.base_options().has_model_file() ? 1 : 0) +
                         (options.has_model_file_with_metadata() ? 1 : 0);
  if (num_input_models != 1) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found %d.",
                        num_input_models),
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  if (options.max_results() == 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "Invalid `max_results` option: value must be != 0",
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  if (options.class_name_whitelist_size() > 0 &&
      options.class_name_blacklist_size() > 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "`class_name_whitelist` and `class_name_blacklist` are mutually "
        "exclusive options.",
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  if (options.num_threads() == 0 || options.num_threads() < -1) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "`num_threads` must be greater than 0 or equal to -1.",
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

absl::Status ImageClassifier::Init(
    std::unique_ptr<ImageClassifierOptions> options) {
  // Set options.
  options_ = std::move(options);

  // Perform pre-initialization actions (by default, sets the process engine for
  // image pre-processing to kLibyuv as a sane default).
  RETURN_IF_ERROR(PreInit());

  // Sanity check and set inputs and outputs.
  RETURN_IF_ERROR(CheckAndSetInputs());

  // ImageClassifier assumes that all output tensors share the same
  // classification option.
  postprocessors_.reserve(engine_->OutputCount(engine_->interpreter()));
  for (int i = 0; i < engine_->OutputCount(engine_->interpreter()); i++) {
    ASSIGN_OR_RETURN(auto processor,
                     CreatePostprocessor(engine_.get(), {i}, *options_));
    postprocessors_.emplace_back(std::move(processor));
  }
  return absl::OkStatus();
}

absl::Status ImageClassifier::PreInit() {
  SetProcessEngine(FrameBufferUtils::ProcessEngine::kLibyuv);
  return absl::OkStatus();
}

StatusOr<ClassificationResult> ImageClassifier::Classify(
    const FrameBuffer& frame_buffer) {
  BoundingBox roi;
  roi.set_width(frame_buffer.dimension().width);
  roi.set_height(frame_buffer.dimension().height);
  return Classify(frame_buffer, roi);
}

StatusOr<ClassificationResult> ImageClassifier::Classify(
    const FrameBuffer& frame_buffer, const BoundingBox& roi) {
  return InferWithFallback(frame_buffer, roi);
}

StatusOr<ClassificationResult> ImageClassifier::Postprocess(
    const std::vector<const TfLiteTensor*>& output_tensors,
    const FrameBuffer& /*frame_buffer*/, const BoundingBox& /*roi*/) {
  ClassificationResult result;
  std::vector<std::pair<int, float>> score_pairs;

  for (int i = 0; i < engine_->OutputCount(engine_->interpreter()); ++i) {
    auto* classifications = result.add_classifications();
    RETURN_IF_ERROR(postprocessors_.at(i)->Postprocess(classifications));
  }

  return result;
}

absl::StatusOr<std::unique_ptr<processor::ClassificationPostprocessor>>
ImageClassifier::CreatePostprocessor(
    core::TfLiteEngine* engine, const std::initializer_list<int> output_indices,
    const ImageClassifierOptions& options) {
  auto new_option = std::make_unique<processor::ClassificationOptions>();
  new_option->set_display_names_locale(options.display_names_locale());
  new_option->set_max_results(options.max_results());
  new_option->set_score_threshold(options.score_threshold());
  new_option->mutable_class_name_allowlist()->Assign(
      options.class_name_whitelist().begin(),
      options.class_name_whitelist().end());
  new_option->mutable_class_name_denylist()->Assign(
      options.class_name_blacklist().begin(),
      options.class_name_blacklist().end());
  return processor::ClassificationPostprocessor::Create(engine, output_indices,
                                                        std::move(new_option));
}

}  // namespace vision
}  // namespace task
}  // namespace tflite
