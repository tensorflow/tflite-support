/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/proto/base_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/text/bert_clu_annotator.h"
#include "tensorflow_lite_support/cc/task/text/clu_annotator.h"
#include "tensorflow_lite_support/cc/task/text/proto/bert_clu_annotator_options.pb.h"
#include "tensorflow_lite_support/cc/utils/jni_utils.h"

namespace {

using ::tflite::support::StatusOr;
using ::tflite::support::utils::GetExceptionClassNameForStatusCode;
using ::tflite::support::utils::kInvalidPointer;
using ::tflite::support::utils::ThrowException;
using ::tflite::task::core::BaseOptions;
using ::tflite::task::text::BertCluAnnotatorOptions;
using ::tflite::task::text::clu::BertCluAnnotator;
using ::tflite::task::text::clu::CluAnnotator;

// Creates a BertCluAnnotatorOptions proto based on the Java class.
BertCluAnnotatorOptions ConvertJavaBertCluAnnotatorProtoOptions(
    JNIEnv* env, jobject java_bert_clu_annotator_options,
    jlong base_options_handle) {
  static jclass bert_clu_annotator_options_class = env->FindClass(
      "org/tensorflow/lite/task/text/bertclu/"
      "BertCluAnnotator$BertCluAnnotatorOptions");
  static jmethodID max_history_turns_method_id = env->GetMethodID(
      bert_clu_annotator_options_class, "getMaxHistoryTurns", "()I");
  static jmethodID domain_threshold_method_id = env->GetMethodID(
      bert_clu_annotator_options_class, "getDomainThreshold", "()F");
  static jmethodID intent_threshold_method_id = env->GetMethodID(
      bert_clu_annotator_options_class, "getIntentThreshold", "()F");
  static jmethodID categorical_slot_threshold_method_id = env->GetMethodID(
      bert_clu_annotator_options_class, "getCategoricalSlotThreshold", "()F");
  static jmethodID noncategorical_slot_threshold_method_id =
      env->GetMethodID(bert_clu_annotator_options_class,
                       "getNoncategoricalSlotThreshold", "()F");
  BertCluAnnotatorOptions proto_options;

  if (base_options_handle != kInvalidPointer) {
    // proto_options will free the previous base_options and set the new one.
    proto_options.set_allocated_base_options(
        reinterpret_cast<BaseOptions*>(base_options_handle));
  }
  proto_options.set_max_history_turns(env->CallIntMethod(
      java_bert_clu_annotator_options, max_history_turns_method_id));
  proto_options.set_domain_threshold(env->CallFloatMethod(
      java_bert_clu_annotator_options, domain_threshold_method_id));
  proto_options.set_intent_threshold(env->CallFloatMethod(
      java_bert_clu_annotator_options, intent_threshold_method_id));
  proto_options.set_categorical_slot_threshold(env->CallFloatMethod(
      java_bert_clu_annotator_options, categorical_slot_threshold_method_id));
  proto_options.set_noncategorical_slot_threshold(
      env->CallFloatMethod(java_bert_clu_annotator_options,
                           noncategorical_slot_threshold_method_id));

  return proto_options;
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_task_text_bertclu_BertCluAnnotator_deinitJni(
    JNIEnv* env, jobject thiz, jlong native_handle) {
  delete reinterpret_cast<CluAnnotator*>(native_handle);
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_task_text_bertclu_BertCluAnnotator_initJniWithByteBuffer(
    JNIEnv* env, jclass thiz, jobject bert_clu_annotator_options,
    jobject model_buffer, jlong base_options_handle) {
  BertCluAnnotatorOptions proto_options =
      ConvertJavaBertCluAnnotatorProtoOptions(env, bert_clu_annotator_options,
                                              base_options_handle);
  proto_options.mutable_base_options()->mutable_model_file()->set_file_content(
      static_cast<char*>(env->GetDirectBufferAddress(model_buffer)),
      static_cast<size_t>(env->GetDirectBufferCapacity(model_buffer)));

  tflite::support::StatusOr<std::unique_ptr<CluAnnotator>> clu_annotator =
      BertCluAnnotator::CreateFromOptions(proto_options);
  if (clu_annotator.ok()) {
    return reinterpret_cast<jlong>(clu_annotator->release());
  } else {
    ThrowException(
        env, GetExceptionClassNameForStatusCode(clu_annotator.status().code()),
        "Error occurred when initializing BertCluAnnotator: %s",
        clu_annotator.status().message().data());
    return kInvalidPointer;
  }
}

// TODO(b/234066484): Implement an `annotateNative` method.
