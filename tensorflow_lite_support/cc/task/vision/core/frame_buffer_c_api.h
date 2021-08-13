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
#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_FRAME_BUFFER_C_API_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_FRAME_BUFFER_C_API_H_

#include <stdint.h>

// --------------------------------------------------------------------------
/// C Struct for FrameBuffer.


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum Format { kRGBA, kRGB, kNV12, kNV21, kYV12, kYV21, kGRAY, kUNKNOWN};
enum Orientation {
  kTopLeft = 1,
  kTopRight = 2,
  kBottomRight = 3,
  kBottomLeft = 4,
  kLeftTop = 5,
  kRightTop = 6,
  kRightBottom = 7,
  kLeftBottom = 8
};

struct Dimension {
  // The width dimension in pixel unit.
  int width;
  // The height dimension in pixel unit.
  int height;
};

struct Plane {
  const uint8_t* buffer;
  
  struct Stride {
    // The row stride in bytes. This is the distance between the start pixels of
    // two consecutive rows in the image.
    int row_stride_bytes;
    // This is the distance between two consecutive pixel values in a row of
    // pixels in bytes. It may be larger than the size of a single pixel to
    // account for interleaved image data or padded formats.
    int pixel_stride_bytes;
  } stride;
};

struct FrameBuffer {
  enum Format format;
  enum Orientation orientation;
  struct Dimension dimension;
  struct Plane plane;
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_VISION_FRAME_BUFFER_C_API_H_
