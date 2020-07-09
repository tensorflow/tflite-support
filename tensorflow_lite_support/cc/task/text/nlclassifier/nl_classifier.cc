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

#include "tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h"

#include "absl/algorithm/container.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/utils/common_utils.h"

namespace tflite {
namespace support {
namespace task {
namespace text {
namespace nlclassifier {

using ::absl::StatusCode;
using ::flatbuffers::Offset;
using ::flatbuffers::Vector;
using ::tflite::TensorMetadata;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::task::core::Dequantize;
using ::tflite::support::task::core::GetStringAtIndex;
using ::tflite::support::task::core::PopulateTensor;
using ::tflite::support::utils::LoadVocabFromBuffer;

const NLClassifierOptions& NLClassifier::GetOptions() const { return options_; }

void NLClassifier::SetOptions(const NLClassifierOptions& options) {
  options_ = options;
}

void NLClassifier::SetLabelsVector(
    std::unique_ptr<std::vector<std::string>> labels_vector) {
  labels_vector_ = std::move(labels_vector);
}

std::vector<core::Category> NLClassifier::Classify(const std::string& text) {
  // The NLClassifier implementation for Preprocess() and Postprocess() never
  // returns errors: just call value().
  return Infer(text).value();
}

absl::Status NLClassifier::Preprocess(
    const std::vector<TfLiteTensor*>& input_tensors, const std::string& input) {
  PopulateTensor(
      input,
      FindTensorWithNameOrIndex(
          input_tensors, GetMetadataExtractor()->GetInputTensorMetadata(),
          options_.input_tensor_name, options_.input_tensor_index));
  return absl::OkStatus();
}

StatusOr<std::vector<core::Category>> NLClassifier::Postprocess(
    const std::vector<const TfLiteTensor*>& output_tensors,
    const std::string& /*input*/) {
  auto scores = FindTensorWithNameOrIndex(
      output_tensors, GetMetadataExtractor()->GetOutputTensorMetadata(),
      options_.output_score_tensor_name, options_.output_score_tensor_index);
  auto labels_tensor = FindTensorWithNameOrIndex(
      output_tensors, GetMetadataExtractor()->GetInputTensorMetadata(),
      options_.output_label_tensor_name, options_.output_label_tensor_index);

  bool use_index_as_labels =
      (labels_vector_ == nullptr) && (labels_tensor == nullptr);
  // Some models output scores with transposed shape [1, categories]
  int categories =
      scores->dims->size == 2 ? scores->dims->data[1] : scores->dims->data[0];

  std::vector<core::Category> predictions;
  predictions.reserve(categories);

  bool should_dequantize = scores->type == kTfLiteUInt8 ||
                           scores->type == kTfLiteInt8 ||
                           scores->type == kTfLiteInt16;
  for (int index = 0; index < categories; index++) {
    std::string label;
    if (use_index_as_labels) {
      label = std::to_string(index);
    } else if (labels_vector_ == nullptr) {
      label = GetStringAtIndex(labels_tensor, index);
    } else {
      label = (*labels_vector_)[index];
    }
    if (should_dequantize) {
      predictions.emplace_back(label, Dequantize(*scores, index));
    } else {
      predictions.emplace_back(label,
                               scores->type == kTfLiteFloat32
                                   ? GetTensorData<float>(scores)[index]
                                   : GetTensorData<double>(scores)[index]);
    }
  }

  return predictions;
}

absl::Status NLClassifier::CheckStatusAndSetOptions(
    const NLClassifierOptions& options, NLClassifier* nl_classifier) {
  nl_classifier->SetOptions(options);
  // input tensor should be type STRING
  auto input_tensor = FindTensorWithNameOrIndex(
      nl_classifier->GetInputTensors(),
      nl_classifier->GetMetadataExtractor()->GetInputTensorMetadata(),
      options.input_tensor_name, options.input_tensor_index);
  if (input_tensor == nullptr) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("No input tensor found with name ",
                     options.input_tensor_name, " or at index ",
                     options.input_tensor_index),
        TfLiteSupportStatus::kInputTensorNotFoundError);
  }
  if (input_tensor->type != kTfLiteString) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("Type mismatch for input tensor ", input_tensor->name,
                     ". Requested STRING, got ",
                     TfLiteTypeGetName(input_tensor->type), "."),
        TfLiteSupportStatus::kInvalidInputTensorTypeError);
  }

  // output score tensor should be type
  // UINT8/INT8/INT16(quantized) or FLOAT32/FLOAT64(dequantized)
  std::vector<const TfLiteTensor*> output_tensors =
      nl_classifier->GetOutputTensors();
  const Vector<Offset<TensorMetadata>>* output_tensor_metadatas =
      nl_classifier->GetMetadataExtractor()->GetOutputTensorMetadata();

  const auto scores = FindTensorWithNameOrIndex(
      output_tensors, output_tensor_metadatas, options.output_score_tensor_name,
      options.output_score_tensor_index);
  if (scores == nullptr) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("No output score tensor found with name ",
                     options.output_score_tensor_name, " or at index ",
                     options.output_score_tensor_index),
        TfLiteSupportStatus::kOutputTensorNotFoundError);
  }
  static constexpr TfLiteType valid_types[] = {
      kTfLiteUInt8, kTfLiteInt8, kTfLiteInt16, kTfLiteFloat32, kTfLiteFloat64};
  if (!absl::c_linear_search(valid_types, scores->type)) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("Type mismatch for score tensor ", scores->name,
                     ". Requested one of these types: "
                     "INT8/UINT8/INT16/FLOAT32/FLOAT64, got ",
                     TfLiteTypeGetName(scores->type), "."),
        TfLiteSupportStatus::kInvalidOutputTensorTypeError);
  }

  // Extract associated label file from output score tensor if one exists, a
  // well-formatted metadata should have same number of tensors with the model.
  if (output_tensor_metadatas &&
      output_tensor_metadatas->size() == output_tensors.size()) {
    for (const auto& metadata : *output_tensor_metadatas) {
      if (metadata->name() &&
          metadata->name()->string_view() == options.output_score_tensor_name) {
        const auto associated_files = metadata->associated_files();
        if (associated_files && associated_files->size() >= 0 &&
            associated_files->Get(0)->name()) {
          StatusOr<absl::string_view> label_buffer =
              nl_classifier->GetMetadataExtractor()->GetAssociatedFile(
                  associated_files->Get(0)->name()->str());
          if (label_buffer.ok()) {
            nl_classifier->SetLabelsVector(
                absl::make_unique<std::vector<std::string>>(LoadVocabFromBuffer(
                    label_buffer.value().data(), label_buffer.value().size())));
          }
        }
      }
    }
  }

  // output label tensor should be type STRING if the one exists
  auto labels = FindTensorWithNameOrIndex(
      output_tensors, output_tensor_metadatas, options.output_label_tensor_name,
      options.output_label_tensor_index);
  if (labels != nullptr && labels->type != kTfLiteString) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat("Type mismatch for label tensor ", scores->name,
                     ". Requested STRING, got ",
                     TfLiteTypeGetName(scores->type), "."),
        TfLiteSupportStatus::kInvalidOutputTensorTypeError);
  }
  return absl::OkStatus();
}

StatusOr<std::unique_ptr<NLClassifier>> NLClassifier::CreateNLClassifier(
    const char* model_buffer_data, size_t model_buffer_size,
    const NLClassifierOptions& options,
    std::unique_ptr<tflite::OpResolver> resolver) {
  std::unique_ptr<NLClassifier> nl_classifier;
  ASSIGN_OR_RETURN(
      nl_classifier,
      core::TaskAPIFactory::CreateFromBuffer<NLClassifier>(
          model_buffer_data, model_buffer_size, std::move(resolver)));
  RETURN_IF_ERROR(CheckStatusAndSetOptions(options, nl_classifier.get()));
  return std::move(nl_classifier);
}

StatusOr<std::unique_ptr<NLClassifier>> NLClassifier::CreateNLClassifier(
    const std::string& path_to_model, const NLClassifierOptions& options,
    std::unique_ptr<tflite::OpResolver> resolver) {
  std::unique_ptr<NLClassifier> nl_classifier;
  ASSIGN_OR_RETURN(nl_classifier,
                   core::TaskAPIFactory::CreateFromFile<NLClassifier>(
                       path_to_model, std::move(resolver)));
  RETURN_IF_ERROR(CheckStatusAndSetOptions(options, nl_classifier.get()));
  return std::move(nl_classifier);
}

StatusOr<std::unique_ptr<NLClassifier>> NLClassifier::CreateNLClassifier(
    int fd, const NLClassifierOptions& options,
    std::unique_ptr<tflite::OpResolver> resolver) {
  std::unique_ptr<NLClassifier> nl_classifier;
  ASSIGN_OR_RETURN(nl_classifier,
                   core::TaskAPIFactory::CreateFromFileDescriptor<NLClassifier>(
                       fd, std::move(resolver)));
  RETURN_IF_ERROR(CheckStatusAndSetOptions(options, nl_classifier.get()));
  return std::move(nl_classifier);
}

}  // namespace nlclassifier
}  // namespace text
}  // namespace task
}  // namespace support
}  // namespace tflite
