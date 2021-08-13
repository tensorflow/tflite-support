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

#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_cpp_c_utils.h"
#include "absl/strings/str_format.h"

using FrameBufferCPP = ::tflite::task::vision::FrameBuffer;

std::unique_ptr<FrameBufferCPP> CreateCPPFrameBuffer(const struct FrameBuffer *frame_buffer) {

  std::unique_ptr<FrameBufferCPP> cpp_frame_buffer;

  if (frame_buffer->format == kRGB) {
    cpp_frame_buffer =
    tflite::task::vision::CreateFromRgbRawBuffer(frame_buffer->plane.buffer, {frame_buffer->dimension.width, frame_buffer->dimension.height});
  } else if (frame_buffer->format == kRGBA) {
    cpp_frame_buffer =
    tflite::task::vision::CreateFromRgbaRawBuffer(frame_buffer->plane.buffer, {frame_buffer->dimension.width, frame_buffer->dimension.height});
  } else if (frame_buffer->format == kGRAY) {
    cpp_frame_buffer =
    tflite::task::vision::CreateFromGrayRawBuffer(frame_buffer->plane.buffer, {frame_buffer->dimension.width, frame_buffer->dimension.height});
  }else {
    return nullptr;
  }
  return cpp_frame_buffer;
}

