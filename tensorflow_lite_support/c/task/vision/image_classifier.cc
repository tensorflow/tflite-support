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

#include "absl/strings/string_view.h"
#include "absl/strings/str_format.h"
#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_classifier_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/c/task/vision/utils/frame_buffer_cpp_c_utils.h"

using ::tflite::support::StatusOr;
using ClassificationResultCPP = ::tflite::task::vision::ClassificationResult;
using ClassificationsCPP = ::tflite::task::vision::Classifications;
using ClassCPP = ::tflite::task::vision::Class;
using BoundingBoxCPP = ::tflite::task::vision::BoundingBox;
using ImageClassifierCPP = ::tflite::task::vision::ImageClassifier;
using ImageClassifierOptionsCPP = ::tflite::task::vision::ImageClassifierOptions;
using FrameBufferCPP = ::tflite::task::vision::FrameBuffer;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct ImageClassifier {
  std::unique_ptr<ImageClassifierCPP> impl;
};

struct ImageClassifierOptions {
  std::unique_ptr<ImageClassifierOptionsCPP> impl;
};

ImageClassifierOptions* ImageClassifierOptionsCreate() {
  return new ImageClassifierOptions{.impl = 
                                    std::unique_ptr<ImageClassifierOptionsCPP> 
                                    (new ImageClassifierOptionsCPP)};
}

void ImageClassifierOptionsSetModelFilePath(ImageClassifierOptions* options,
    const char* model_path) {
  options->impl->mutable_base_options()->mutable_model_file()
      ->set_file_name(model_path);
}
void ImageClassifierOptionsSetDisplayNamesLocal(
    ImageClassifierOptions* options, char *display_names_locale) {
  options->impl->set_display_names_locale(display_names_locale);
}

void ImageClassifierOptionsSetMaxResults(ImageClassifierOptions* options, 
    int max_results) {
  options->impl->set_max_results(max_results);
}

void ImageClassifierOptionsSetScoreThreshold(ImageClassifierOptions* options, 
    float score_threshold) {
  options->impl->set_score_threshold(score_threshold);
}

void ImageClassifierOptionsSetNumThreads(ImageClassifierOptions* options, 
    int num_threads) {
  options->impl->mutable_base_options()->mutable_compute_settings()
      ->mutable_tflite_settings()->mutable_cpu_settings()
      ->set_num_threads(num_threads);
}

void ImageClassifierOptionsAddClassNameWhiteList(
    ImageClassifierOptions* options, char* class_name) {
  options->impl->add_class_name_whitelist(class_name);
}

void ImageClassifierOptionsAddClassNameBlackList(
  ImageClassifierOptions* options, char* class_name) {
  options->impl->add_class_name_blacklist(class_name);
}

void ImageClassifierOptionsDelete(
    ImageClassifierOptions* options) { delete options; }


ImageClassifierOptions* ImageClassifierOptionsDefault() {
  ImageClassifierOptions* image_classifier_options = 
      ImageClassifierOptionsCreate();
  ImageClassifierOptionsSetMaxResults(image_classifier_options, 5);
  ImageClassifierOptionsSetScoreThreshold(image_classifier_options, 0);
  
  return image_classifier_options;
}


ImageClassifier* ImageClassifierFromOptions(
    const ImageClassifierOptions* options) {
  auto classifier_status = 
      ImageClassifierCPP::CreateFromOptions(*(options->impl));

  if (classifier_status.ok()) {
    return new ImageClassifier{.impl = std::unique_ptr<ImageClassifierCPP>(
                                    dynamic_cast<ImageClassifierCPP*>(
                                        classifier_status.value().release()))};
  } else {
    return nullptr; 
  }
}

ImageClassifier* ImageClassifierFromFile(const char* model_path) {
  ImageClassifierOptions *default_options = ImageClassifierOptionsDefault();
  ImageClassifierOptionsSetModelFilePath(default_options, model_path);
  return ImageClassifierFromOptions(default_options);
}

ClassificationResult* GetClassificationResultCStruct(
    const ClassificationResultCPP classification_result_cpp) {
  auto* c_classifications = 
      new Classifications[classification_result_cpp.classifications_size()];

  for(int head = 0; head < classification_result_cpp.classifications_size(); ++head) {
    const ClassificationsCPP& classifications = 
        classification_result_cpp.classifications(head);
    c_classifications[head].head_index = head;
    
    auto* c_classes = new Class[classifications.classes_size()];
    c_classifications->size = classifications.classes_size();

    for (int rank = 0; rank < classifications.classes_size(); ++rank) {
      const ClassCPP& classification = classifications.classes(rank);
      c_classes[rank].index = classification.index();
      c_classes[rank].score = classification.score();
      
      if (classification.has_class_name())
        c_classes[rank].class_name = 
            strdup(classification.class_name().c_str());
      else
        c_classes[rank].class_name = nullptr;
      
      if (classification.has_display_name())
        c_classes[rank].display_name = 
            strdup(classification.display_name().c_str());
      else
        c_classes[rank].display_name = nullptr;
    }
    c_classifications[head].classes = c_classes;
  }
  
  ClassificationResult *c_classification_result = new ClassificationResult;
  c_classification_result->classifications = c_classifications;
  c_classification_result->size = 
      classification_result_cpp.classifications_size();
  
  return c_classification_result;
}

struct ClassificationResult* ImageClassifierClassify(
    const ImageClassifier* classifier,
    const struct FrameBuffer* frame_buffer) {
  std::unique_ptr<FrameBufferCPP> cpp_frame_buffer = CreateCPPFrameBuffer(frame_buffer);

  if (cpp_frame_buffer == nullptr)
    return nullptr;
    
  StatusOr<ClassificationResultCPP> classification_result_cpp = 
      classifier->impl->Classify(*cpp_frame_buffer);
  
  if (!classification_result_cpp.ok())
    return nullptr;

  ClassificationResult *c_classification_result = 
      GetClassificationResultCStruct(
          ClassificationResultCPP(classification_result_cpp.value()));

  return c_classification_result;
}

struct ClassificationResult* ImageClassifierClassifyWithBoundingBox(
    const ImageClassifier* classifier, const struct FrameBuffer* frame_buffer,
    const struct BoundingBox* roi) {
    
    BoundingBoxCPP cc_roi;
    cc_roi.set_origin_x(roi->origin_x);
    cc_roi.set_origin_y(roi->origin_y);
    cc_roi.set_width(roi->width);
    cc_roi.set_height(roi->height);
  
    std::unique_ptr<FrameBufferCPP> cpp_frame_buffer = 
        CreateCPPFrameBuffer(frame_buffer);
    StatusOr<ClassificationResultCPP> classification_result_cpp = 
        classifier->impl->Classify(*cpp_frame_buffer, cc_roi);
    
    if (!classification_result_cpp.ok())
      return nullptr;

    ClassificationResult *c_classification_result = 
        GetClassificationResultCStruct(
            ClassificationResultCPP(classification_result_cpp.value()));

    return c_classification_result;
}

void ImageClassifierDelete(ImageClassifier* classifier) { delete classifier; }


#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
