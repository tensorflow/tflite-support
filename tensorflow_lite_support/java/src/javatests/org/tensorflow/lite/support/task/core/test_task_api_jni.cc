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

#include <jni.h>

#include "third_party/absl/status/status.h"

// Test api object to perform a simple add function.
//
// The two constructors are no-ops for demonstration, for a real Api object,
// provide implementation to initialize it with/without files accordingly.
class TestApi {
 public:
  // Initializes the api without any buffer.
  TestApi() = default;

  // Loads the two files passed from Java and initialize the API.
  TestApi(const char *buffer1, size_t buffer_size1, const char *buffer2,
          size_t buffer_size2) {}

  int Add(int i1, int i2) const { return i1 + i2; }
};

extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_support_task_core_BaseTaskApi_deinitJni(
    JNIEnv *env, jobject thiz, jlong nativeHandle) {
  delete reinterpret_cast<TestApi *>(nativeHandle);
}

extern "C" JNIEXPORT jint JNICALL
Java_org_tensorflow_lite_support_task_core_TestTaskApi_addNative(
    JNIEnv *env, jobject thiz, jlong native_handle, jint i1, jint i2) {
  auto *test_api = reinterpret_cast<TestApi *>(native_handle);
  return test_api->Add(i1, i2);
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_support_task_core_TestTaskApi_initJni(JNIEnv *env,
                                                               jclass thiz) {
  auto api = absl::make_unique<TestApi>();
  return reinterpret_cast<jlong>(api.release());
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_support_task_core_TestTaskApi_initJniWithByteBuffers(
    JNIEnv *env, jclass thiz, jobjectArray buffers) {
  auto buffer1 = env->GetObjectArrayElement(buffers, 0);
  auto buffer2 = env->GetObjectArrayElement(buffers, 1);

  const char *buff1_ptr =
      static_cast<char *>(env->GetDirectBufferAddress(buffer1));
  auto buff1_size = static_cast<size_t>(env->GetDirectBufferCapacity(buffer1));

  const char *buff2_ptr =
      static_cast<char *>(env->GetDirectBufferAddress(buffer2));
  auto buff2_size = static_cast<size_t>(env->GetDirectBufferCapacity(buffer2));

  auto api =
      absl::make_unique<TestApi>(buff1_ptr, buff1_size, buff2_ptr, buff2_size);
  return reinterpret_cast<jlong>(api.release());
}
