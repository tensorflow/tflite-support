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
#ifndef TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_VISION_COMMON_H_
#define TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_VISION_COMMON_H_

#include <stdint.h>

// --------------------------------------------------------------------------
/// Common  C APIs and Structs for Vision Tasks.
//

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Holds the region of interest used for image classification.
struct BoundingBox {
  // The X coordinate of the top-left corner, in pixels.
  int32_t origin_x;

  // The Y coordinate of the top-left corner, in pixels.
  int32_t origin_y;

  // The width of the bounding box, in pixels.
  int32_t width;

  // The height of the bounding box, in pixels.
  int32_t height;
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_C_TASK_VISION_VISION_COMMON_H_