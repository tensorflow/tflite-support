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

#include "tensorflow_lite_support/cc/task/vision/classification_result_c_api.h"

#include <memory>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

void ImageClassifierClassificationResultDelete(struct ClassificationResult* classification_result) {
  for (int head = 0; head < classification_result->size; ++head) {
    
    for (int rank = 0; rank < classification_result->classifications->size; ++rank) {
      // `strdup` obtains memory using `malloc` and the memory needs to be
      // released using `free`.
      free(classification_result->classifications->classes[rank].display_name);
      free(classification_result->classifications->classes[rank].class_name);
    }
    
    delete[] classification_result->classifications->classes;
  }
  delete[] classification_result->classifications;
  delete classification_result;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus