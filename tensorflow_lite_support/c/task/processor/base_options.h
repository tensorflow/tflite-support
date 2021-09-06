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
#ifndef TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_BASE_OPTIONS_H_
#define TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_BASE_OPTIONS_H_

#include <stdint.h>

// --------------------------------------------------------------------------
/// Common  C APIs and Structs for Vision Tasks.
//

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus


typedef struct TfLiteCpuSettings {
int num_threads;
} TfLiteCpuSettings;

typedef struct TfLiteTflSettings {
TfLiteCpuSettings cpu_settings;
} TfLiteTflSettings;

typedef struct TfLiteComputeSettings {

TfLiteTflSettings tflite_settings;
} TfLiteComputeSettings;

typedef struct TfLiteExternalFile {
    const char* file_path;
    //Additional option for byte data when it's supported.
} TfLiteExternalFile;

// Holds the region of interest used for image classification.
typedef struct TfLiteBaseOptions {
  // The X coordinate of the top-left corner, in pixels.
  TfLiteExternalFile model_file;
  TfLiteComputeSettings compute_settings;

} TfLiteBaseOptions;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_C_TASK_PROCESSOR_BASE_OPTIONS_H_
