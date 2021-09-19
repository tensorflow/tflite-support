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
#ifndef TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_CLASSIFICATION_OPTIONS_H_
#define TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_CLASSIFICATION_OPTIONS_H_

#include <stdint.h>

// Defines C Struct for Classification Options Shared by All Classification
// Tasks.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Holds pointer to array of C strings and length for looping through the array.
typedef struct TfLiteStringArrayOption {
  // Length of list. length can be used to loop through list.
  int length;

  // Array of C strings.
  const char** list;
} TfLiteStringArrayOption;

// Holds settings for any single classification task.
// TODO(prianka): change `white/blacklist` to `allow/denylist`
typedef struct TfLiteClassificationOptions {
  // Optional blacklist of class names. If non NULL, classifications whose
  // class name is in this set will be filtered out. Duplicate or unknown
  // class names are ignored. Mutually exclusive with class_name_whitelist.
  TfLiteStringArrayOption class_name_blacklist;

  // Optional whitelist of class names. If non-empty, classifications whose
  // class name is not in this set will be filtered out. Duplicate or unknown
  // class names are ignored. Mutually exclusive with class_name_blacklist.
  TfLiteStringArrayOption class_name_whitelist;

  const char* display_names_local;
  int max_results;
  float score_threshold;
} TfLiteClassificationOptions;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_CLASSIFICATION_OPTIONS_H_
