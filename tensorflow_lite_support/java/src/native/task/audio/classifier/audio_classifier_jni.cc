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

#include <memory>
#include <string>

#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/audio/audio_classifier.h"
#include "tensorflow_lite_support/cc/task/audio/proto/audio_classifier_options.proto.h"
#include "tensorflow_lite_support/cc/task/audio/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/utils/jni_utils.h"

namespace {

using ::tflite::support::StatusOr;
using ::tflite::support::utils::kAssertionError;
using ::tflite::support::utils::kInvalidPointer;
using ::tflite::support::utils::StringListToVector;
using ::tflite::support::utils::ThrowException;
using ::tflite::task::audio::AudioClassifier;
using ::tflite::task::audio::AudioClassifierOptions;

// Creates an AudioClassifierOptions proto based on the Java class.
AudioClassifierOptions ConvertToProtoOptions(JNIEnv* env,
                                             jobject java_options) {
  AudioClassifierOptions proto_options;
  jclass java_options_class = env->FindClass(
      "org/tensorflow/lite/task/audio/classifier/"
      "AudioClassifier$AudioClassifierOptions");

  jmethodID display_names_locale_id = env->GetMethodID(
      java_options_class, "getDisplayNamesLocale", "()Ljava/lang/String;");
  jstring display_names_locale = static_cast<jstring>(
      env->CallObjectMethod(java_options, display_names_locale_id));
  const char* pchars = env->GetStringUTFChars(display_names_locale, nullptr);
  proto_options.set_display_names_locale(pchars);
  env->ReleaseStringUTFChars(display_names_locale, pchars);

  jmethodID max_results_id =
      env->GetMethodID(java_options_class, "getMaxResults", "()I");
  jint max_results = env->CallIntMethod(java_options, max_results_id);
  proto_options.set_max_results(max_results);

  jmethodID is_score_threshold_set_id =
      env->GetMethodID(java_options_class, "getIsScoreThresholdSet", "()Z");
  jboolean is_score_threshold_set =
      env->CallBooleanMethod(java_options, is_score_threshold_set_id);
  if (is_score_threshold_set) {
    jmethodID score_threshold_id =
        env->GetMethodID(java_options_class, "getScoreThreshold", "()F");
    jfloat score_threshold =
        env->CallFloatMethod(java_options, score_threshold_id);
    proto_options.set_score_threshold(score_threshold);
  }

  jmethodID allow_list_id = env->GetMethodID(
      java_options_class, "getLabelAllowList", "()Ljava/util/List;");
  jobject allow_list = env->CallObjectMethod(java_options, allow_list_id);
  auto allow_list_vector = StringListToVector(env, allow_list);
  for (const auto& class_name : allow_list_vector) {
    proto_options.add_class_name_whitelist(class_name);
  }

  jmethodID deny_list_id = env->GetMethodID(
      java_options_class, "getLabelDenyList", "()Ljava/util/List;");
  jobject deny_list = env->CallObjectMethod(java_options, deny_list_id);
  auto deny_list_vector = StringListToVector(env, deny_list);
  for (const auto& class_name : deny_list_vector) {
    proto_options.add_class_name_blacklist(class_name);
  }

  return proto_options;
}

jlong CreateAudioClassifierFromOptions(JNIEnv* env,
                                       const AudioClassifierOptions& options) {
  StatusOr<std::unique_ptr<AudioClassifier>> audio_classifier_or =
      AudioClassifier::CreateFromOptions(options);
  if (audio_classifier_or.ok()) {
    // Deletion is handled at deinitJni time.
    return reinterpret_cast<jlong>(audio_classifier_or->release());
  } else {
    ThrowException(env, kAssertionError,
                   "Error occurred when initializing AudioClassifier: %s",
                   audio_classifier_or.status().message().data());
  }
  return kInvalidPointer;
}

extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_task_vision_classifier_ImageClassifier_deinitJni(
    JNIEnv* env, jobject thiz, jlong native_handle) {
  delete reinterpret_cast<AudioClassifier*>(native_handle);
}

// Creates an ImageClassifier instance from the model file descriptor.
// file_descriptor_length and file_descriptor_offset are optional. Non-possitive
// values will be ignored.
extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_task_audio_classifier_AudioClassifier_initJniWithModelFdAndOptions(
    JNIEnv* env, jclass thiz, jint file_descriptor,
    jlong file_descriptor_length, jlong file_descriptor_offset,
    jobject java_options) {
  AudioClassifierOptions proto_options =
      ConvertToProtoOptions(env, java_options);
  auto file_descriptor_meta = proto_options.mutable_base_options()
                                  ->mutable_model_file()
                                  ->mutable_file_descriptor_meta();
  file_descriptor_meta->set_fd(file_descriptor);
  if (file_descriptor_length > 0) {
    file_descriptor_meta->set_length(file_descriptor_length);
  }
  if (file_descriptor_offset > 0) {
    file_descriptor_meta->set_offset(file_descriptor_offset);
  }
  return CreateAudioClassifierFromOptions(env, proto_options);
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_task_audio_classifier_AudioClassifier_initJniWithByteBuffer(
    JNIEnv* env, jclass thiz, jobject model_buffer, jobject java_options) {
  AudioClassifierOptions proto_options =
      ConvertToProtoOptions(env, java_options);
  // External proto generated header does not overload `set_file_content` with
  // string_view, therefore GetMappedFileBuffer does not apply here.
  // Creating a std::string will cause one extra copying of data. Thus, the
  // most efficient way here is to set file_content using char* and its size.
  proto_options.mutable_base_options()->mutable_model_file()->set_file_content(
      static_cast<char*>(env->GetDirectBufferAddress(model_buffer)),
      static_cast<size_t>(env->GetDirectBufferCapacity(model_buffer)));
  return CreateAudioClassifierFromOptions(env, proto_options);
}
}  // namespace
