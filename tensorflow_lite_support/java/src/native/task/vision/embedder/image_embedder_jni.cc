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

#include <memory>

#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/proto/base_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/processor/proto/embedding.pb.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/image_embedder.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_embedder_options.pb.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/cc/utils/jni_utils.h"
#include "tensorflow_lite_support/java/src/native/task/vision/jni_utils.h"

namespace tflite {
namespace task {
// To be provided by a link-time library
extern std::unique_ptr<OpResolver> CreateOpResolver();

}  // namespace task
}  // namespace tflite

namespace {

using ::tflite::support::StatusOr;
using ::tflite::support::utils::ConvertVectorToArrayList;
using ::tflite::support::utils::CreateFloatArray;
using ::tflite::support::utils::CreateByteArray;
using ::tflite::support::utils::GetExceptionClassNameForStatusCode;
using ::tflite::support::utils::kInvalidPointer;
using ::tflite::support::utils::ThrowException;
using ::tflite::task::core::BaseOptions;
using ::tflite::task::processor::FeatureVector;
using ::tflite::task::processor::Embedding;
using ::tflite::task::processor::EmbeddingResult;
using ::tflite::task::vision::BoundingBox;
using ::tflite::task::vision::FrameBuffer;
using ::tflite::task::vision::ImageEmbedder;
using ::tflite::task::vision::ImageEmbedderOptions;

// Creates an ImageEmbedderOptions proto based on the Java class.
ImageEmbedderOptions ConvertToProtoOptions(JNIEnv* env, jobject java_options,
                                            jlong base_options_handle) {
  ImageEmbedderOptions proto_options;

  if (base_options_handle != kInvalidPointer) {
    // proto_options will free the previous base_options and set the new one.
    proto_options.set_allocated_base_options(
        reinterpret_cast<BaseOptions*>(base_options_handle));
  }

  jclass java_options_class = env->FindClass(
      "org/tensorflow/lite/task/vision/embedder/"
      "ImageEmbedder$ImageEmbedderOptions");

  jmethodID l2_normalize_id =
      env->GetMethodID(java_options_class, "getL2Normalize", "()Z");
  jint l2_normalize = env->CallBooleanMethod(java_options, l2_normalize_id);
  proto_options.set_l2_normalize(l2_normalize);

  jmethodID quantization_id =
      env->GetMethodID(java_options_class, "getQuantize", "()Z");
  jint quantization = env->CallBooleanMethod(java_options, quantization_id);
  proto_options.set_quantize(quantization);

  return proto_options;
}

jobject ConvertToEmbedResults(JNIEnv* env, const EmbeddingResult& results) {
  // jclass and factory create of Embedding.
  jclass embedding_class =
      env->FindClass("org/tensorflow/lite/task/processor/Embedding");
  jmethodID embedding_create =
      env->GetStaticMethodID(embedding_class, "create",
                             "([Lorg/tensorflow/lite/task/processor/FeatureVector;I)"
                             "Lorg/tensorflow/lite/task/processor/Embedding;");

  // jclass and factory create of FeatureVector.
  jclass feature_vector_class =
      env->FindClass("org/tensorflow/lite/task/processor/FeatureVector");
  jmethodID feature_vector_create =
      env->GetStaticMethodID(feature_vector_class, "create",
                             "([FB)Lorg/tensorflow/lite/task/processor/FeatureVector;");

  
  return ConvertVectorToArrayList(
      env, results.embeddings().begin(), results.embeddings().end(),
      [env, embedding_class, embedding_create, feature_vector_class,
       feature_vector_create](const Embedding& embedding) {
        jobject jfeature_vector = nullptr;
        if (embedding.has_feature_vector()) {
          const FeatureVector& feature_vector = embedding.feature_vector();
          if (feature_vector.value_float_size() > 0) {
            // jfloatArray jfloat_array = CreateFloatArray(env, feature_vector.value_float());
            jfloatArray jfloat_array = CreateFloatArray(
                env, reinterpret_cast<const jfloat*>(feature_vector.value_float().data()),
                feature_vector.value_float().size());
            jfeature_vector = env->CallStaticObjectMethod(
                feature_vector_class, feature_vector_create, jfloat_array);
            env->DeleteLocalRef(jfloat_array);
          } else if (feature_vector.has_value_string()) {
            jbyteArray jvalue_string = CreateByteArray(
                env, reinterpret_cast<const jbyte*>(feature_vector.value_string().data()),
                feature_vector.value_string().size());
            jfeature_vector = env->CallStaticObjectMethod(
                feature_vector_class, feature_vector_create, jvalue_string);
            env->DeleteLocalRef(jvalue_string);
          }
        }
        jobject jembedding = env->CallStaticObjectMethod(
            embedding_class, embedding_create, jfeature_vector, embedding.output_index());
        env->DeleteLocalRef(jfeature_vector);
        return jembedding;
      });
}

jlong CreateImageEmbedderFromOptions(JNIEnv* env,
                                     const ImageEmbedderOptions& options) {
  StatusOr<std::unique_ptr<ImageEmbedder>> image_embedder_or =
      ImageEmbedder::CreateFromOptions(options,
                                       tflite::task::CreateOpResolver());
  if (image_embedder_or.ok()) {
    return reinterpret_cast<jlong>(image_embedder_or->release());
  } else {
    ThrowException(
        env,
        GetExceptionClassNameForStatusCode(image_embedder_or.status().code()),
        "Error occurred when initializing ImagEmbedder: %s",
        image_embedder_or.status().message().data());
    return kInvalidPointer;
  }
}

}  // namespace

extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_task_vision_embedder_ImageEmbedder_deinitJni(
    JNIEnv* env, jobject thiz, jlong native_handle) {
  delete reinterpret_cast<ImageEmbedder*>(native_handle);
}

// Creates an ImageEmbedder instance from the model file descriptor.
// file_descriptor_length and file_descriptor_offset are optional. Non-positive
// values will be ignored.
extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_task_vision_embedder_ImageEmbedder_initJniWithModelFdAndOptions(
    JNIEnv* env, jclass thiz, jint file_descriptor,
    jlong file_descriptor_length, jlong file_descriptor_offset,
    jobject java_options, jlong base_options_handle) {
  ImageEmbedderOptions proto_options =
      ConvertToProtoOptions(env, java_options, base_options_handle);
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

  return CreateImageEmbedderFromOptions(env, proto_options);
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_task_vision_embedder_ImageEmbedder_initJniWithByteBuffer(
    JNIEnv* env, jclass thiz, jobject model_buffer, jobject java_options,
    jlong base_options_handle) {
  ImageEmbedderOptions proto_options =
      ConvertToProtoOptions(env, java_options, base_options_handle);
  proto_options.mutable_base_options()->mutable_model_file()->set_file_content(
      static_cast<char*>(env->GetDirectBufferAddress(model_buffer)),
      static_cast<size_t>(env->GetDirectBufferCapacity(model_buffer)));

  return CreateImageEmbedderFromOptions(env, proto_options);
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_tensorflow_lite_task_vision_embedder_ImageEmbedder_embedNative(
    JNIEnv* env, jclass thiz, jlong native_handle, jlong frame_buffer_handle,
    jintArray jroi) {
  auto* embedder = reinterpret_cast<ImageEmbedder*>(native_handle);
  // frame_buffer will be deleted after inference is done in
  // base_vision_api_jni.cc.
  auto* frame_buffer = reinterpret_cast<FrameBuffer*>(frame_buffer_handle);

  int* roi_array = env->GetIntArrayElements(jroi, 0);
  BoundingBox roi;
  roi.set_origin_x(roi_array[0]);
  roi.set_origin_y(roi_array[1]);
  roi.set_width(roi_array[2]);
  roi.set_height(roi_array[3]);
  env->ReleaseIntArrayElements(jroi, roi_array, 0);

  auto results_or = embedder->Embed(*frame_buffer, roi);
  if (results_or.ok()) {
    return ConvertToEmbedResults(env, results_or.value());
  } else {
    ThrowException(
        env, GetExceptionClassNameForStatusCode(results_or.status().code()),
        "Error occurred when embedding the image: %s",
        results_or.status().message().data());
    return nullptr;
  }
}
