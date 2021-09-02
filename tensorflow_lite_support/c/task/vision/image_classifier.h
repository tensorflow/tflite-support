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
#ifndef TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_IMAGE_CLASSIFIER_H_
#define TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_IMAGE_CLASSIFIER_H_

#include <stdint.h>

#include "tensorflow_lite_support/c/task/processor/bounding_box.h"
#include "tensorflow_lite_support/c/task/processor/classification_result.h"
#include "tensorflow_lite_support/c/task/vision/core/frame_buffer.h"

// --------------------------------------------------------------------------
/// C API for ImageClassifiier.
///
/// The API leans towards simplicity and uniformity instead of convenience, as
/// most usage will be by language-specific wrappers. It provides largely the
/// same set of functionality as that of the C++ TensorFlow Lite
/// `ImageClassifier` API, but is useful for shared libraries where having
/// a stable ABI boundary is important.
///
/// Usage with Model File Path:
/// <pre><code>
/// // Create the model
/// ImageClassifier* classifier =
///   ImageClassifierFromFile("/path/to/model.tflite");
///
/// TfLiteFrameBuffer frame_buffer = { Initialize }
///
/// Create the model
/// TfLiteClassificationResult *classification_result =
///   ImageClassifierClassify(classifier, &frame_buffer,);
///
/// // Dispose of the API object.
/// ImageClassifierDelete(classifier);
///
/// Advanced Usage with Options:
/// <pre><code>
/// // Create the interpreter options
/// ImageClassifierOptions *options = ImageClassifierOptionsCreate();
/// ImageClassifierOptionsSetScoreThreshold(options, 0.5);
/// ImageClassifierOptionsSetMaxResults(options, 3);
/// ImageClassifierOptionsSetModelFilePath(options, model_path);
///
/// TfLiteFrameBuffer frame_buffer = { Initialize }
///
/// // Create the model
/// ImageClassifier* classifier =
///   ImageClassifierFromOptions(options);
/// struct TfLiteClassificationResult *classification_result =
///   ImageClassifierClassify(classifier, &frame_buffer,);
///
/// // Dispose of the API object.
/// ImageClassifierDelete(classifier);
/// Advanced Usage with Options:
/// <pre><code>
/// // Create the interpreter options
/// ImageClassifierOptions *options = ImageClassifierOptionsCreate();
/// ImageClassifierOptionsSetScoreThreshold(options, 0.5);
/// ImageClassifierOptionsSetMaxResults(options, 3);
/// ImageClassifierOptionsSetModelFilePath(options, model_path);
///
/// TfLiteFrameBuffer frame_buffer = { Initialize }
///
/// // Create the model
/// ImageClassifier* classifier =
///   ImageClassifierFromOptions(options);
/// TfLiteClassificationResult *classification_result =
///   ImageClassifierClassify(classifier, &frame_buffer,);
///
/// // Dispose of the API object.
/// ImageClassifierDelete(classifier);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct TfLiteImageClassifier TfLiteImageClassifier;

typedef struct TfLiteImageClassifierOptions TfLiteImageClassifierOptions;

// Creates and returns the ImageClassifierOptions
TfLiteImageClassifierOptions* TfLiteImageClassifierOptionsCreate();

// Creates and returns the ImageClassifierOptions
void TfLiteImageClassifierOptionsSetModelFilePath(
    TfLiteImageClassifierOptions* options, const char* model_path);

// Sets the Display names local option in the ImageClassifierOptions
void TfLiteImageClassifierOptionsSetDisplayNamesLocal(
    TfLiteImageClassifierOptions* image_classifier_options,
    char* display_names_locale);

// Sets the maximum number of classification results in the encapsulated
// image classifier options.
extern void TfLiteImageClassifierOptionsSetMaxResults(
    TfLiteImageClassifierOptions* image_classifier_options, int max_results);

// Sets the score threshold of classification results to be
// returned after classification in the ImageClassifierOptions.
// Score threshold, overrides the ones provided in the model metadata
// (if any).Only results predicted with a confidence threshold of greater than
// the score threshold are returned. The value of score threshold
// should be between 0 and 1.
void TfLiteImageClassifierOptionsSetScoreThreshold(
    TfLiteImageClassifierOptions* image_classifier_options,
    float score_threshold);

// Sets the number of threads used for classification in the encapsulated
// ImageClassifierOptions
void TfLiteImageClassifierOptionsSetNumThreads(
    TfLiteImageClassifierOptions* image_classifier_options, int num_threads);

// Adds a class name to the list of class names to be whitelisted. If you have
// more than one class names to be whitelisted, consider calling this method
// repeatedly in a loop. If you set atleast one class name into the whitelist
// using this option, classifications whose class names which are not added
// into the white list will be filtered out. Duplicate or unknown
// class names are ignored. Mutually exclusive with blacklisted class names set
// using void ImageClassifierOptionsAddClassNameBlackList(
// ImageClassifierOptions* image_classifier_options, char* class_name);
void TfLiteImageClassifierOptionsAddClassNameWhiteList(
    TfLiteImageClassifierOptions* image_classifier_options, char* class_name);

// Adds a class name to the list of class names to be black listed. If you have
// more than one class names to be black listed, consider calling this method
// repeatedly in a loop. Classifications whose class are added into the black
// list will be filtered out. Duplicate or unknown class names are ignored.
// Mutually exclusive with white listed class names set
// using void ImageClassifierOptionsAddClassNameWhiteList(
// ImageClassifierOptions* image_classifier_options, char* class_name);
void TfLiteImageClassifierOptionsAddClassNameBlackList(
    TfLiteImageClassifierOptions* image_classifier_options, char* class_name);

// Disposes off the ImageClassifierOptions.
void TfLiteImageClassifierOptionsDelete(TfLiteImageClassifierOptions* options);

// Creates ImageClassifier from options. When using this function to instantiate
// Image Classifer, the model path should be set using
// ImageClassifierOptionsSetModelFilePath(ImageClassifierOptions* options,
//                                          const char* model_path)
// returns nullptr if the file doesn't exist or is not a well formatted
// TFLite model path.
TfLiteImageClassifier* TfLiteImageClassifierFromOptions(
    const TfLiteImageClassifierOptions* options);

// Creates ImaegeClassifier from model path and default options, returns nullptr
// if the file doesn't exist or is not a well formatted TFLite model path.
// This image classifier created using this function defaults to s score
// threshold of 0, max_results of 5. In order to override these, use:
// ImageClassifier* ImageClassifierFromOptions(
//   const ImageClassifierOptions* options) instead.
TfLiteImageClassifier* TfLiteImageClassifierFromFile(const char* model_path);

// Invokes the encapsulated TFLite model and classifies the frame_buffer.
TfLiteClassificationResult* TfLiteImageClassifierClassify(
    const TfLiteImageClassifier* classifier,
    const TfLiteFrameBuffer* frame_buffer);

// Invokes the encapsulated TFLite model and classifies the
// region of the frame_buffer specified by the bounding box.
// Same as above, except that the classification is performed based on the
// input region of interest. Cropping according to this region of interest is
// prepended to the pre-processing operations.
TfLiteClassificationResult* TfLiteImageClassifierClassifyWithRoi(
    const TfLiteImageClassifier* classifier,
    const TfLiteFrameBuffer* frame_buffer, const TfLiteBoundingBox* roi);

// Disposes off the image classifier
void TfLiteImageClassifierDelete(TfLiteImageClassifier* classifier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_IMAGE_CLASSIFIER_H_
