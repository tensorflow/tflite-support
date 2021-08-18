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
#ifndef TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_CLASSIFICATION_RESULT_C_API_H_
#define TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_CLASSIFICATION_RESULT_C_API_H_

// --------------------------------------------------------------------------
// Defines C structure for Classification Results and associated helper methods
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct Class {
  int index;
  float score;
  char *display_name;
  char *class_name;
  
};

struct Classifications {
 
  int head_index;
  int size;
  struct Class *classes;
  
};
  
struct ClassificationResult {
  
  int size;
  struct Classifications *classifications;
  
};

// Frees up the classification result structure.
extern void ImageClassifierClassificationResultDelete(struct ClassificationResult* classification_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_CLASSIFICATION_RESULT_C_API_H_
