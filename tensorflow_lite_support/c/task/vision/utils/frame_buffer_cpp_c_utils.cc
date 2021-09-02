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

#include "tensorflow_lite_support/c/task/vision/utils/frame_buffer_cpp_c_utils.h"

#include "absl/strings/str_format.h"

using FrameBufferCpp = ::tflite::task::vision::FrameBuffer;

std::unique_ptr<FrameBufferCpp> CreateCppFrameBuffer(
    const TfLiteFrameBuffer* frame_buffer) {
  FrameBufferCpp::Format frame_buffer_format =
      FrameBufferCpp::Format((*frame_buffer).format);

  auto cpp_frame_buffer = tflite::task::vision::CreateFromRawBuffer(
      frame_buffer->buffer,
      {frame_buffer->dimension.width, frame_buffer->dimension.height},
      frame_buffer_format);

  if (!cpp_frame_buffer.ok()) {
    return nullptr;
  }

  return std::unique_ptr<FrameBufferCpp>(
      dynamic_cast<FrameBufferCpp*>(cpp_frame_buffer.value().release()));
}
