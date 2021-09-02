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

#include "tensorflow_lite_support/c/task/vision/image_classifier.h"

#include <memory>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "tensorflow_lite_support/c/task/vision/utils/frame_buffer_cpp_c_utils.h"
#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"
#include "tensorflow_lite_support/cc/task/vision/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_classifier_options_proto_inc.h"

namespace {
using ::tflite::support::StatusOr;
using ClassificationResultCpp = ::tflite::task::vision::ClassificationResult;
using ClassificationsCpp = ::tflite::task::vision::Classifications;
using ClassCpp = ::tflite::task::vision::Class;
using BoundingBoxCpp = ::tflite::task::vision::BoundingBox;
using ImageClassifierCpp = ::tflite::task::vision::ImageClassifier;
using ImageClassifierOptionsCpp =
    ::tflite::task::vision::ImageClassifierOptions;
using FrameBufferCpp = ::tflite::task::vision::FrameBuffer;
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct TfLiteImageClassifier {
  std::unique_ptr<ImageClassifierCpp> impl;
};

struct TfLiteImageClassifierOptions {
  std::unique_ptr<ImageClassifierOptionsCpp> impl;
};

TfLiteImageClassifierOptions* TfLiteImageClassifierOptionsCreate() {
  return new TfLiteImageClassifierOptions{
      .impl = std::unique_ptr<ImageClassifierOptionsCpp>(
          new ImageClassifierOptionsCpp)};
}

void TfLiteImageClassifierOptionsSetModelFilePath(
    TfLiteImageClassifierOptions* options, const char* model_path) {
  options->impl->mutable_base_options()->mutable_model_file()->set_file_name(
      model_path);
}
void TfLiteImageClassifierOptionsSetDisplayNamesLocal(
    TfLiteImageClassifierOptions* options, char* display_names_locale) {
  options->impl->set_display_names_locale(display_names_locale);
}

void TfLiteImageClassifierOptionsSetMaxResults(
    TfLiteImageClassifierOptions* options, int max_results) {
  options->impl->set_max_results(max_results);
}

void TfLiteImageClassifierOptionsSetScoreThreshold(
    TfLiteImageClassifierOptions* options, float score_threshold) {
  options->impl->set_score_threshold(score_threshold);
}

void TfLiteImageClassifierOptionsSetNumThreads(
    TfLiteImageClassifierOptions* options, int num_threads) {
  options->impl->mutable_base_options()
      ->mutable_compute_settings()
      ->mutable_tflite_settings()
      ->mutable_cpu_settings()
      ->set_num_threads(num_threads);
}

void TfLiteImageClassifierOptionsAddClassNameWhiteList(
    TfLiteImageClassifierOptions* options, char* class_name) {
  options->impl->add_class_name_whitelist(class_name);
}

void TfLiteImageClassifierOptionsAddClassNameBlackList(
    TfLiteImageClassifierOptions* options, char* class_name) {
  options->impl->add_class_name_blacklist(class_name);
}

void TfLiteImageClassifierOptionsDelete(TfLiteImageClassifierOptions* options) {
  delete options;
}

TfLiteImageClassifierOptions* TfLiteImageClassifierOptionsDefault() {
  TfLiteImageClassifierOptions* image_classifier_options =
      TfLiteImageClassifierOptionsCreate();
  TfLiteImageClassifierOptionsSetMaxResults(image_classifier_options, 5);
  TfLiteImageClassifierOptionsSetScoreThreshold(image_classifier_options, 0);

  return image_classifier_options;
}

TfLiteImageClassifier* TfLiteImageClassifierFromOptions(
    const TfLiteImageClassifierOptions* options) {
  auto classifier_status =
      ImageClassifierCpp::CreateFromOptions(*(options->impl));

  if (classifier_status.ok()) {
    return new TfLiteImageClassifier{
        .impl = std::unique_ptr<ImageClassifierCpp>(
            dynamic_cast<ImageClassifierCpp*>(
                classifier_status.value().release()))};
  } else {
    return nullptr;
  }
}

TfLiteImageClassifier* TfLiteImageClassifierFromFile(const char* model_path) {
  TfLiteImageClassifierOptions* default_options =
      TfLiteImageClassifierOptionsDefault();
  TfLiteImageClassifierOptionsSetModelFilePath(default_options, model_path);
  return TfLiteImageClassifierFromOptions(default_options);
}

TfLiteClassificationResult* GetClassificationResultCStruct(
    const ClassificationResultCpp classification_result_cpp) {
  auto* c_classifications =
      new TfLiteClassifications[classification_result_cpp
                                    .classifications_size()];

  for (int head = 0; head < classification_result_cpp.classifications_size();
       ++head) {
    const ClassificationsCpp& classifications =
        classification_result_cpp.classifications(head);
    c_classifications[head].head_index = head;

    auto* c_categories = new TfLiteCategory[classifications.classes_size()];
    c_classifications->size = classifications.classes_size();

    for (int rank = 0; rank < classifications.classes_size(); ++rank) {
      const ClassCpp& classification = classifications.classes(rank);
      c_categories[rank].index = classification.index();
      c_categories[rank].score = classification.score();

      if (classification.has_class_name())
        c_categories[rank].class_name =
            strdup(classification.class_name().c_str());
      else
        c_categories[rank].class_name = nullptr;

      if (classification.has_display_name())
        c_categories[rank].display_name =
            strdup(classification.display_name().c_str());
      else
        c_categories[rank].display_name = nullptr;
    }
    c_classifications[head].categories = c_categories;
  }

  TfLiteClassificationResult* c_classification_result =
      new TfLiteClassificationResult;
  c_classification_result->classifications = c_classifications;
  c_classification_result->size =
      classification_result_cpp.classifications_size();

  return c_classification_result;
}

TfLiteClassificationResult* TfLiteImageClassifierClassify(
    const TfLiteImageClassifier* classifier,
    const TfLiteFrameBuffer* frame_buffer) {
  std::unique_ptr<FrameBufferCpp> cpp_frame_buffer =
      CreateCppFrameBuffer(frame_buffer);

  if (cpp_frame_buffer == nullptr) return nullptr;

  StatusOr<ClassificationResultCpp> classification_result_cpp =
      classifier->impl->Classify(*cpp_frame_buffer);

  if (!classification_result_cpp.ok()) return nullptr;

  TfLiteClassificationResult* c_classification_result =
      GetClassificationResultCStruct(
          ClassificationResultCpp(classification_result_cpp.value()));

  return c_classification_result;
}

TfLiteClassificationResult* TfLiteImageClassifierClassifyWithRoi(
    const TfLiteImageClassifier* classifier,
    const TfLiteFrameBuffer* frame_buffer, const TfLiteBoundingBox* roi) {
  BoundingBoxCpp cc_roi;
  cc_roi.set_origin_x(roi->origin_x);
  cc_roi.set_origin_y(roi->origin_y);
  cc_roi.set_width(roi->width);
  cc_roi.set_height(roi->height);

  std::unique_ptr<FrameBufferCpp> cpp_frame_buffer =
      CreateCppFrameBuffer(frame_buffer);
  StatusOr<ClassificationResultCpp> classification_result_cpp =
      classifier->impl->Classify(*cpp_frame_buffer, cc_roi);

  if (!classification_result_cpp.ok()) return nullptr;

  TfLiteClassificationResult* c_classification_result =
      GetClassificationResultCStruct(
          ClassificationResultCpp(classification_result_cpp.value()));

  return c_classification_result;
}

void TfLiteImageClassifierDelete(TfLiteImageClassifier* classifier) {
  delete classifier;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
