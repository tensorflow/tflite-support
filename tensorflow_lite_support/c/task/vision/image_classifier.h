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

#include "tensorflow_lite_support/c/task/processor/base_options.h"
#include "tensorflow_lite_support/c/task/processor/bounding_box.h"
#include "tensorflow_lite_support/c/task/processor/classification_options.h"
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
/// TfLiteImageClassifierOptions options = {0};
///   options.base_options.model_file.file_path = "/path/to/model.tflite";
///   TfLiteImageClassifier* image_classifier =
///       TfLiteImageClassifierFromOptions(&options);
///
/// Classify an image
/// TfLiteFrameBuffer frame_buffer = { Initialize with image data }
///
/// TfLiteClassificationResult* classification_result =
///       TfLiteImageClassifierClassify(image_classifier, &frame_buffer);
///
/// // Dispose of the API object.
/// TfLiteImageClassifierDelete(image_classifier);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct TfLiteImageClassifier TfLiteImageClassifier;

typedef struct TfLiteImageClassifierOptions {
  TfLiteClassificationOptions classification_options;
  TfLiteBaseOptions base_options;

} TfLiteImageClassifierOptions;

// Creates TfLiteImageClassifier from options. When using this function to
// instantiate Image Classifer, the model path should be set using
// TfLiteImageClassifierOptionsSetModelFilePath(TfLiteImageClassifierOptions*
// options,
//                                          const char* model_path)
// returns nullptr if the file doesn't exist or is not a well formatted
// TFLite model path.
TfLiteImageClassifier* TfLiteImageClassifierFromOptions(
    const TfLiteImageClassifierOptions* options);

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
