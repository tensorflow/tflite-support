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

#ifndef TENSORFLOW_LITE_SUPPORT_CC_UTILS_JNI_UTILS_H_
#define TENSORFLOW_LITE_SUPPORT_CC_UTILS_JNI_UTILS_H_

#include <jni.h>

#include <functional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace tflite {
namespace support {
namespace utils {

const char kIllegalStateException[] = "java/lang/IllegalStateException";

// Check if t is nullptr, throw IllegalStateException if it is.
// Used to verify different types of jobjects are correctly created from jni.
template <typename T>
T CheckNotNull(JNIEnv* env, T&& t) {
  if (t == nullptr) {
    env->ThrowNew(env->FindClass(kIllegalStateException), "");
    return nullptr;
  }
  return std::forward<T>(t);
}

// Convert a vector<T> into an Java ArrayList using a converter.
template <typename T>
jobject ConvertVectorToArrayList(JNIEnv* env, const std::vector<T>& results,
                                 std::function<jobject(T)> converter) {
  jclass array_list_class = env->FindClass("java/util/ArrayList");
  jmethodID array_list_ctor =
      env->GetMethodID(array_list_class, "<init>", "(I)V");
  jint initial_capacity = static_cast<jint>(results.size());
  jobject array_list_object =
      env->NewObject(array_list_class, array_list_ctor, initial_capacity);
  jmethodID array_list_add_method =
      env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

  for (const auto& ans : results) {
    env->CallBooleanMethod(array_list_object, array_list_add_method,
                           converter(ans));
  }
  return array_list_object;
}

std::string JStringToString(JNIEnv* env, jstring jstr);

std::vector<std::string> StringListToVector(JNIEnv* env, jobject list_object);

// Gets a mapped file buffer from a java object representing a file.
absl::string_view GetMappedFileBuffer(JNIEnv* env, const jobject& file_buffer);
}  // namespace utils
}  // namespace support
}  // namespace tflite
#endif  // TENSORFLOW_LITE_SUPPORT_CC_UTILS_JNI_UTILS_H_
