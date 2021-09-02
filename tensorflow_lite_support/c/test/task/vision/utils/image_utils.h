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
#ifndef TENSORFLOW_LITE_SUPPORT_C_TEST_TASK_VISION_IMAGE_UTILS_H_
#define TENSORFLOW_LITE_SUPPORT_C_TEST_TASK_VISION_IMAGE_UTILS_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
// Image data with pixels stored as a row-major flattened array.
// Channels can be:
// 1 : grayscale
// 3 : RGB, interleaved
// 4 : RGBA, interleaved
typedef struct CImageData {
  uint8_t* pixel_data;
  int width;
  int height;
  int channels;
} CImageData;

// Decodes image file and returns the corresponding image if no error
// occurred. If decoding succeeded, the caller must manage deletion of the
// underlying pixel data using `ImageDataFree`.
// Supports a wide range of image formats, listed in `stb_image/stb_image.h`.
CImageData CDecodeImageFromFile(const char* file_name);

// Releases image pixel data memory.
void CImageDataFree(CImageData* image_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_C_TEST_TASK_VISION_IMAGE_UTILS_H_
