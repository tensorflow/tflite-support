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

#include <jni.h>

#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/utils/jni_utils.h"
#include "tensorflow_lite_support/java/src/native/task/vision/jni_utils.h"

namespace {

using ::tflite::support::utils::kAssertionError;
using ::tflite::support::utils::kInvalidPointer;
using ::tflite::support::utils::ThrowException;
using ::tflite::task::vision::CreateFrameBuffer;
using ::tflite::task::vision::FrameBuffer;

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_task_vision_core_BaseVisionTaskApi_createFrameBufferFromByteBuffer(
    JNIEnv* env, jclass thiz, jobject jimage_byte_buffer, jint width,
    jint height, jint jorientation, jint jcolor_space_type) {
  auto frame_buffer_or = CreateFrameBuffer(
      env, jimage_byte_buffer, width, height, jorientation, jcolor_space_type);
  if (frame_buffer_or.ok()) {
    return reinterpret_cast<jlong>(frame_buffer_or->release());
  } else {
    ThrowException(env, kAssertionError,
                   "Error occurred when creating FrameBuffer: %s",
                   frame_buffer_or.status().message().data());
    return kInvalidPointer;
  }
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_task_vision_core_BaseVisionTaskApi_createFrameBufferFromBytes(
    JNIEnv* env, jclass thiz, jbyteArray jimage_bytes, jint width, jint height,
    jint jorientation, jint jcolor_space_type, jlongArray jbyte_array_handle) {
  auto frame_buffer_or =
      CreateFrameBuffer(env, jimage_bytes, width, height, jorientation,
                        jcolor_space_type, jbyte_array_handle);
  if (frame_buffer_or.ok()) {
    return reinterpret_cast<jlong>(frame_buffer_or->release());
  } else {
    ThrowException(env, kAssertionError,
                   "Error occurred when creating FrameBuffer: %s",
                   frame_buffer_or.status().message().data());
    return kInvalidPointer;
  }
}

extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_task_vision_core_BaseVisionTaskApi_deleteFrameBuffer(
    JNIEnv* env, jobject thiz, jlong frame_buffer_handle,
    jlong byte_array_handle, jbyteArray jbyte_array) {
  delete reinterpret_cast<FrameBuffer*>(frame_buffer_handle);
  jbyte* bytes_ptr = reinterpret_cast<jbyte*>(byte_array_handle);
  if (bytes_ptr != NULL) {
    env->ReleaseByteArrayElements(jbyte_array, bytes_ptr, /*mode=*/0);
  }
}

}  // namespace
