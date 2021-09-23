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

#include "tensorflow_lite_support/cc/task/vision/image_embedder.h"

#include <algorithm>

#include "external/com_google_absl/absl/container/node_hash_set.h"
#include "external/com_google_absl/absl/memory/memory.h"
#include "external/com_google_absl/absl/status/status.h"
#include "external/com_google_absl/absl/strings/str_format.h"
#include "external/com_google_absl/absl/strings/string_view.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_utils.h"

namespace tflite {
namespace task {
namespace vision {

namespace {
using ::absl::StatusCode;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::core::TaskAPIFactory;

// Performs actual cosine similarity computation.
template <typename T>
tflite::support::StatusOr<double> ComputeCosineSimilarity(const T* u,
                                                          const T* v,
                                                          int num_elements) {
  if (num_elements <= 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "Cannot compute cosine similarity on empty feature vectors",
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  double dot_product = 0.0;
  double norm_u = 0.0;
  double norm_v = 0.0;
  for (int i = 0; i < num_elements; ++i) {
    dot_product += u[i] * v[i];
    norm_u += u[i] * u[i];
    norm_v += v[i] * v[i];
  }
  if (norm_u <= 0.0 || norm_v <= 0.0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "Cannot compute cosine similarity on feature vector with 0 norm",
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  return dot_product / std::sqrt(norm_u * norm_v);
}

}  // namespace

/* static */
tflite::support::StatusOr<double> ImageEmbedder::CosineSimilarity(
    const FeatureVector& u, const FeatureVector& v) {
  if (u.has_value_string() && v.has_value_string()) {
    if (u.value_string().size() != v.value_string().size()) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Cannot compute cosine similarity on quantized "
                          "feature vectors of different sizes (%d vs %d)",
                          u.value_string().size(), v.value_string().size()),
          TfLiteSupportStatus::kInvalidArgumentError);
    }
    return ComputeCosineSimilarity(
        reinterpret_cast<const int8_t*>(&u.value_string()[0]),
        reinterpret_cast<const int8_t*>(&v.value_string()[0]),
        u.value_string().size());
  }
  if (!u.has_value_string() && !v.has_value_string()) {
    if (u.value_float_size() != v.value_float_size()) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Cannot compute cosine similarity on float "
                          "feature vectors of different sizes (%d vs %d)",
                          u.value_float_size(), v.value_float_size()),
          TfLiteSupportStatus::kInvalidArgumentError);
    }
    return ComputeCosineSimilarity(
        u.value_float().data(), v.value_float().data(), u.value_float().size());
  }
  return CreateStatusWithPayload(
      StatusCode::kInvalidArgument,
      "Cannot compute cosine similarity between quantized and float "
      "feature vectors",
      TfLiteSupportStatus::kInvalidArgumentError);
}

/* static */
absl::Status ImageEmbedder::SanityCheckOptions(
    const ImageEmbedderOptions& options) {
  // Nothing to check.
  return absl::OkStatus();
}

/* static */
tflite::support::StatusOr<std::unique_ptr<ImageEmbedder>>
ImageEmbedder::CreateFromOptions(const ImageEmbedderOptions& options,
                                 std::unique_ptr<tflite::OpResolver> resolver) {
  RETURN_IF_ERROR(SanityCheckOptions(options));

  // Copy options to ensure the ExternalFile-s outlive the constructed object.
  auto options_copy = absl::make_unique<ImageEmbedderOptions>(options);

  ASSIGN_OR_RETURN(
      auto image_embedder,
      TaskAPIFactory::CreateFromExternalFileProto<ImageEmbedder>(
          &options_copy->model_file_with_metadata(), std::move(resolver),
          options_copy->num_threads(), options_copy->compute_settings()));

  RETURN_IF_ERROR(image_embedder->Init(std::move(options_copy)));

  return image_embedder;
}

absl::Status ImageEmbedder::PreInit() {
  SetProcessEngine(FrameBufferUtils::ProcessEngine::kLibyuv);
  return absl::OkStatus();
}

absl::Status ImageEmbedder::PostInit() {
  // Nothing to do.
  return absl::OkStatus();
}

absl::Status ImageEmbedder::Init(
    std::unique_ptr<ImageEmbedderOptions> options) {
  // Set options.
  options_ = std::move(options);

  // Perform pre-initialization actions.
  RETURN_IF_ERROR(PreInit());

  // Sanity check and set inputs and outputs.
  RETURN_IF_ERROR(CheckAndSetInputs());
  RETURN_IF_ERROR(CheckAndSetOutputs());

  // Perform post-initialization actions.
  RETURN_IF_ERROR(PostInit());

  return absl::OkStatus();
}

absl::Status ImageEmbedder::CheckAndSetOutputs() {
  // First, sanity checks on the model itself.
  num_output_layers_ = GetTfLiteEngine()->interpreter()->outputs().size();
  embedding_dimensions_.resize(num_output_layers_);

  int num_quantized_outputs = 0;
  for (int i = 0; i < num_output_layers_; ++i) {
    int output_tensor_index = GetTfLiteEngine()->interpreter()->outputs()[i];
    const TfLiteTensor* output_tensor =
        GetTfLiteEngine()->interpreter()->tensor(output_tensor_index);
    int num_dimensions = output_tensor->dims->size;
    if (num_dimensions == 4) {
      if (output_tensor->dims->data[1] != 1 ||
          output_tensor->dims->data[2] != 1) {
        return CreateStatusWithPayload(
            StatusCode::kInvalidArgument,
            absl::StrFormat("Unexpected WxH sizes for output index %d: got "
                            "%dx%d, expected 1x1.",
                            i, output_tensor->dims->data[2],
                            output_tensor->dims->data[1]),
            TfLiteSupportStatus::kInvalidOutputTensorDimensionsError);
      }
    } else if (num_dimensions != 2) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat(
              "Unexpected number of dimensions for output index %d: got %dD, "
              "expected either 2D (BxN with B=1) or 4D (BxHxWxN with B=1, "
              "W=1, "
              "H=1).",
              i, num_dimensions),
          TfLiteSupportStatus::kInvalidOutputTensorDimensionsError);
    }
    if (output_tensor->dims->data[0] != 1) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("The output array is expected to have a batch size "
                          "of 1. Got %d for output index %d.",
                          output_tensor->dims->data[0], i),
          TfLiteSupportStatus::kInvalidOutputTensorDimensionsError);
    }
    embedding_dimensions_[i] = output_tensor->dims->data[num_dimensions - 1];
    if (output_tensor->type == kTfLiteUInt8) {
      num_quantized_outputs++;
    } else if (output_tensor->type != kTfLiteFloat32) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Type mismatch for output tensor %s. Requested one "
                          "of these types: "
                          "kTfLiteUint8/kTfLiteFloat32, got %s.",
                          output_tensor->name,
                          TfLiteTypeGetName(output_tensor->type)),
          TfLiteSupportStatus::kInvalidOutputTensorTypeError);
    }
  }

  if (num_quantized_outputs > 0 &&
      num_quantized_outputs != num_output_layers_) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Got %d quantized output(s), expected %d (i.e. all "
                        "provided outputs must be quantized).",
                        num_quantized_outputs, num_output_layers_),
        TfLiteSupportStatus::kInvalidOutputTensorTypeError);
  }
  has_uint8_outputs_ = (num_quantized_outputs > 0);

  return absl::OkStatus();
}

tflite::support::StatusOr<EmbeddingResult> ImageEmbedder::Embed(
    const FrameBuffer& frame_buffer) {
  BoundingBox roi;
  roi.set_width(frame_buffer.dimension().width);
  roi.set_height(frame_buffer.dimension().height);
  return Embed(frame_buffer, roi);
}

tflite::support::StatusOr<EmbeddingResult> ImageEmbedder::Embed(
    const FrameBuffer& frame_buffer, const BoundingBox& roi) {
  return InferWithFallback(frame_buffer, roi);
}

tflite::support::StatusOr<EmbeddingResult> ImageEmbedder::Postprocess(
    const std::vector<const TfLiteTensor*>& output_tensors,
    const FrameBuffer& /*frame_buffer*/, const BoundingBox& /*roi*/) {
  EmbeddingResult result;
  for (int i = 0; i < num_output_layers_; ++i) {
    Embedding* embedding = result.add_embeddings();
    embedding->set_output_index(i);
    FeatureVector* feature_vector = embedding->mutable_feature_vector();
    if (has_uint8_outputs_) {
      const uint8* output_data =
          GetTfLiteEngine()->interpreter()->typed_output_tensor<uint8>(i);
      // Get the zero_point and scale parameters from the tensor metadata.
      const int output_tensor_index =
          GetTfLiteEngine()->interpreter()->outputs()[i];
      const TfLiteTensor* output_tensor =
          GetTfLiteEngine()->interpreter()->tensor(output_tensor_index);
      for (int j = 0; j < embedding_dimensions_[i]; ++j) {
        feature_vector->add_value_float(output_tensor->params.scale *
                                        (static_cast<int>(output_data[j]) -
                                         output_tensor->params.zero_point));
      }
    } else {
      const float* output_data =
          GetTfLiteEngine()->interpreter()->typed_output_tensor<float>(i);
      for (int j = 0; j < embedding_dimensions_[i]; ++j) {
        feature_vector->add_value_float(output_data[j]);
      }
    }
  }
  if (options_->l2_normalize()) {
    NormalizeResult(&result);
  }
  if (options_->quantize()) {
    QuantizeResult(&result);
  }

  return result;
}

Embedding ImageEmbedder::GetEmbeddingByIndex(const EmbeddingResult& result,
                                             int output_index) {
  if (output_index < 0 || output_index >= num_output_layers_) {
    return Embedding();
  }
  return result.embeddings(output_index);
}

int ImageEmbedder::GetEmbeddingDimension(int output_index) const {
  if (output_index < 0 || output_index >= num_output_layers_) {
    return -1;
  }
  return embedding_dimensions_[output_index];
}

int ImageEmbedder::GetNumberOfOutputLayers() const {
  return num_output_layers_;
}

void ImageEmbedder::NormalizeFeatureVector(
    FeatureVector* feature_vector) const {
  float squared_l2_norm = 0.0f;
  for (const float val : feature_vector->value_float()) {
    squared_l2_norm += val * val;
  }
  if (squared_l2_norm == 0.0f) {
    return;
  }
  const float inv_l2_norm = 1.0f / std::sqrt(squared_l2_norm);
  for (int i = 0; i < feature_vector->value_float().size(); ++i) {
    feature_vector->set_value_float(
        i, feature_vector->value_float(i) * inv_l2_norm);
  }
}

void ImageEmbedder::NormalizeResult(EmbeddingResult* result) const {
  for (int i = 0; i < num_output_layers_; ++i) {
    FeatureVector* feature_vector =
        result->mutable_embeddings(i)->mutable_feature_vector();
    NormalizeFeatureVector(feature_vector);
  }
}

void ImageEmbedder::QuantizeFeatureVector(FeatureVector* feature_vector) const {
  auto* quantized_values = feature_vector->mutable_value_string();
  quantized_values->resize(feature_vector->value_float().size());
  for (int i = 0; i < feature_vector->value_float().size(); ++i) {
    int value = static_cast<int>(roundf(feature_vector->value_float(i) * 128));
    (*quantized_values)[i] =
        static_cast<char>(std::max(-128, std::min(value, 127)));
  }
}

void ImageEmbedder::QuantizeResult(EmbeddingResult* result) const {
  for (int i = 0; i < num_output_layers_; ++i) {
    FeatureVector* feature_vector =
        result->mutable_embeddings(i)->mutable_feature_vector();
    QuantizeFeatureVector(feature_vector);
    feature_vector->clear_value_float();
  }
}

}  // namespace vision
}  // namespace task
}  // namespace tflite
