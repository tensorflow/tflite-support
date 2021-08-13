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
#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_IMAGE_CLASSIFIER_C_API_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_IMAGE_CLASSIFIER_C_API_H_

#include <stdint.h>

#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer_c_api.h"
#include "tensorflow_lite_support/cc/task/vision/classification_result_c_api.h"
#include "tensorflow_lite_support/cc/task/vision/vision_common_c_api.h"

// --------------------------------------------------------------------------
/// C API for ImageClassifiier.
///
/// The API leans towards simplicity and uniformity instead of convenience, as
/// most usage will be by language-specific wrappers. It provides largely the
/// same set of functionality as that of the C++ TensorFlow Lite
/// `ImageClassifier` API, but is useful for shared libraries where having
/// a stable ABI boundary is important.
///
/// Basic Usage:
/// <pre><code>
/// // Create the model
/// ImageClassifier* classifier =
///   ImageClassifierFromFile("/path/to/model.tflite");
///
/// struct FrameBuffer frame_buffer = { Initialize }
///
/// Create the model
/// struct ClassificationResult *classification_result = ImageClassifierClassify(classifier, &frame_buffer,);
///
/// // Dispose of the API object.
/// ImageClassifierDelete(classifier);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct ImageClassifier ImageClassifier;

typedef struct ImageClassifierOptions ImageClassifierOptions;

// Creates and reeturns the ImageClassifierOptions
extern ImageClassifierOptions* ImageClassifierOptionsCreate();

// Creates and returns the ImageClassifierOptions
extern void ImageClassifierOptionsSetModelFilePath(ImageClassifierOptions* options, const char* model_path);

// Sets the Display names local option in the ImageClassifierOptions
extern void ImageClassifierOptionsSetDisplayNamesLocal(
    ImageClassifierOptions* image_classifier_options, char *display_names_locale);

// Sets the maximum number of classification results to be
// returned in the ImageClassifierOptions
extern void ImageClassifierOptionsSetMaxResults(
    ImageClassifierOptions* image_classifier_options, int max_results);

// Sets the score threshold of classification results to be
// returned after classification in the ImageClassifierOptions.
// Only results predicted with a confidence threshold of greater than
// the score threshold are returned. The value of score threshold
// should be between 0 and 1.
extern void ImageClassifierOptionsSetScoreThreshold(
    ImageClassifierOptions* image_classifier_options, float score_threshold);

extern void ImageClassifierOptionsSetNumThreads(
    ImageClassifierOptions* image_classifier_options, int num_threads);

extern void ImageClassifierOptionsAddClassNameWhiteList(
    ImageClassifierOptions* image_classifier_options, char* class_name);

extern void ImageClassifierOptionsAddClassNameBlackList(
    ImageClassifierOptions* image_classifier_options, char* class_name);

void ImageClassifierOptionsDelete(ImageClassifierOptions* options);


extern const struct ImageClassifierOptions ImageClassifierOptions_default;

// Creates ImageClassifier from options. The model path should be set using
// ImageClassifierOptionsSetModelFilePath(ImageClassifierOptions* options, const char* model_path)
// returns nullptr if the file doesn't exist or is not a well formatted TFLite model path.
extern ImageClassifier* ImageClassifierFromOptions(const ImageClassifierOptions* options);

// Creates ImaegeClassifier from model path and default options, returns nullptr
// if the file doesn't exist or is not a well formatted TFLite model path.
extern ImageClassifier* ImageClassifierFromFile(const char* model_path);

//// Invokes the encapsulated TFLite model and classifies the frame_buffer.
extern struct ClassificationResult* ImageClassifierClassify(
    const ImageClassifier* classifier, const struct FrameBuffer* frame_buffer);

// Invokes the encapsulated TFLite model and classifies the region of the frame_buffer
// specified by the bounding box.
extern struct ClassificationResult* ImageClassifierClassifyWithBoundingBox(
    const ImageClassifier* classifier, const struct FrameBuffer* frame_buffer, const struct BoundingBox* roi);

// Disposes off the image classifier
extern void ImageClassifierDelete(ImageClassifier* classifier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_IMAGE_CLASSIFIER_C_API_H_
