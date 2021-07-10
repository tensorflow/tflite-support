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

#include "tensorflow_lite_support/cc/task/audio/audio_classifier.h"

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/integral_types.h"
#include "tensorflow_lite_support/cc/task/audio/proto/class_proto_inc.h"
#include "tensorflow_lite_support/cc/task/audio/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/task/core/classification_head.h"
#include "tensorflow_lite_support/cc/task/core/label_map_item.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace task {
namespace audio {

namespace {

using ::absl::StatusCode;
using ::tflite::AudioProperties;
using ::tflite::ContentProperties;
using ::tflite::ContentProperties_AudioProperties;
using ::tflite::metadata::ModelMetadataExtractor;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::audio::Class;
using ::tflite::task::audio::ClassificationResult;
using ::tflite::task::core::AssertAndReturnTypedTensor;
using ::tflite::task::core::LabelMapItem;
using ::tflite::task::core::TaskAPIFactory;
using ::tflite::task::core::TfLiteEngine;

StatusOr<const TensorMetadata*> GetInputTensorMetadata(
    const ModelMetadataExtractor& metadata_extractor) {
  if (metadata_extractor.GetModelMetadata() == nullptr ||
      metadata_extractor.GetModelMetadata()->subgraph_metadata() == nullptr) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "Models are assumed to have the ModelMetadata and SubGraphMetadata.",
        TfLiteSupportStatus::kMetadataNotFoundError);
  } else if (metadata_extractor.GetInputTensorCount() != 1) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "Models are assumed to have a single input TensorMetadata.",
        TfLiteSupportStatus::kInvalidNumInputTensorsError);
  }

  const TensorMetadata* metadata = metadata_extractor.GetInputTensorMetadata(0);

  if (metadata == nullptr) {
    // Should never happen.
    return CreateStatusWithPayload(StatusCode::kInternal,
                                   "Input TensorMetadata is null.");
  }

  return metadata;
}

StatusOr<const AudioProperties*> GetAudioProperties(
    const TensorMetadata& tensor_metadata) {
  if (tensor_metadata.content() == nullptr ||
      tensor_metadata.content()->content_properties() == nullptr) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        "Missing audio format metadata in the model metadata.",
        TfLiteSupportStatus::kMetadataNotFoundError);
  }

  ContentProperties type = tensor_metadata.content()->content_properties_type();

  if (type != ContentProperties_AudioProperties) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrCat(
            "Expected AudioProperties for tensor ",
            tensor_metadata.name() ? tensor_metadata.name()->str() : "#0",
            ", got ", EnumNameContentProperties(type), "."),
        TfLiteSupportStatus::kMetadataInvalidContentPropertiesError);
  }

  return tensor_metadata.content()->content_properties_as_AudioProperties();
}

}  // namespace

/* static */
StatusOr<std::unique_ptr<AudioClassifier>> AudioClassifier::CreateFromOptions(
    const AudioClassifierOptions& options,
    std::unique_ptr<tflite::OpResolver> resolver) {
  RETURN_IF_ERROR(SanityCheckOptions(options));

  // Copy options to ensure the ExternalFile outlives the constructed object.
  auto options_copy = absl::make_unique<AudioClassifierOptions>(options);

  ASSIGN_OR_RETURN(auto audio_classifier,
                   TaskAPIFactory::CreateFromBaseOptions<AudioClassifier>(
                       &options_copy->base_options(), std::move(resolver)));

  RETURN_IF_ERROR(audio_classifier->Init(std::move(options_copy)));

  return audio_classifier;
}

/* static */
absl::Status AudioClassifier::SanityCheckOptions(
    const AudioClassifierOptions& options) {
  if (!options.has_base_options()) {
    return CreateStatusWithPayload(StatusCode::kInvalidArgument,
                                   "Missing mandatory `base_options` field",
                                   TfLiteSupportStatus::kInvalidArgumentError);
  }
  if (options.max_results() == 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "Invalid `max_results` option: value must be != 0",
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  if (options.class_name_allowlist_size() > 0 &&
      options.class_name_denylist_size() > 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "`class_name_allowlist` and `class_name_denylist` are mutually "
        "exclusive options.",
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  return absl::OkStatus();
}

absl::Status AudioClassifier::Init(
    std::unique_ptr<AudioClassifierOptions> options) {
  // Set options.
  options_ = std::move(options);
  RETURN_IF_ERROR(SetAudioFormatFromMetadata());
  RETURN_IF_ERROR(CheckAndSetInputs());
  RETURN_IF_ERROR(CheckAndSetOutputs());

  return absl::OkStatus();
}

absl::Status AudioClassifier::SetAudioFormatFromMetadata() {
  const ModelMetadataExtractor* metadata_extractor =
      engine_->metadata_extractor();
  ASSIGN_OR_RETURN(const TensorMetadata* metadata,
                   GetInputTensorMetadata(*metadata_extractor));
  ASSIGN_OR_RETURN(const AudioProperties* props, GetAudioProperties(*metadata));
  if (props == nullptr) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        "Missing audio format metadata in the model metadata.",
        TfLiteSupportStatus::kMetadataNotFoundError);
  }
  audio_format_.channels = props->channels();
  audio_format_.sample_rate = props->sample_rate();
  return absl::OkStatus();
}

tflite::support::StatusOr<AudioBuffer::AudioFormat>
AudioClassifier::GetRequiredAudioFormat() {
  if (audio_format_.channels <= 0 || audio_format_.sample_rate <= 0) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "Missing audio format metadata in the model.",
        TfLiteSupportStatus::kMetadataNotFoundError);
  }
  return audio_format_;
}

absl::Status AudioClassifier::CheckAndSetInputs() {
  const std::vector<TfLiteTensor*> input_tensors = GetInputTensors();
  input_buffer_size_ = 1;
  TfLiteIntArray* dims = input_tensors[0]->dims;
  for (int i = 0; i < dims->size; ++i) {
    if (dims->data[i] < 1) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Invalid size: %d for input tensor dimension: %d.",
                          dims->data[i], i),
          TfLiteSupportStatus::kInvalidInputTensorDimensionsError);
    }
    input_buffer_size_ *= input_tensors[0]->dims->data[i];
  }

  // Check if the input buffer size is divisible by the required audio channels.
  // This needs to be done after loading metadata and input.
  if (input_buffer_size_ % audio_format_.channels != 0) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Model input tensor size (%d) should be a "
                        "multiplier of the number of channels (%d).",
                        input_buffer_size_, audio_format_.channels),
        TfLiteSupportStatus::kMetadataInconsistencyError);
  }
  return absl::OkStatus();
}

// TODO(b/182537114): Extract into a common library to share between audio and
// vision tasks.
absl::Status AudioClassifier::CheckAndSetOutputs() {
  num_outputs_ = TfLiteEngine::OutputCount(engine_->interpreter());

  // Perform sanity checks and extract metadata.
  const ModelMetadataExtractor* metadata_extractor =
      engine_->metadata_extractor();

  const flatbuffers::Vector<flatbuffers::Offset<tflite::TensorMetadata>>*
      output_tensor_metadata = metadata_extractor->GetOutputTensorMetadata();

  // Loop over output tensors metadata, if any.
  // Note: models with no output tensor metadata at all are supported.
  if (output_tensor_metadata != nullptr) {
    int num_output_tensors = output_tensor_metadata->size();

    if (num_outputs_ != num_output_tensors) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Mismatch between number of output tensors (%d) and "
                          "output tensors "
                          "metadata (%d).",
                          num_outputs_, num_output_tensors),
          TfLiteSupportStatus::kMetadataInconsistencyError);
    }

    for (int i = 0; i < num_output_tensors; ++i) {
      const tflite::TensorMetadata* output_tensor =
          output_tensor_metadata->Get(i);

      ASSIGN_OR_RETURN(
          core::ClassificationHead head,
          core::BuildClassificationHead(*metadata_extractor, *output_tensor,
                                        options_->display_names_locale()));

      classification_heads_.emplace_back(std::move(head));
    }
  }

  // If classifier heads are not set, build default ones based on model
  // introspection. This happens if a model with partial or no metadata was
  // provided through the `model_file_with_metadata` options field.
  if (classification_heads_.empty()) {
    classification_heads_.reserve(num_outputs_);
    for (int output_index = 0; output_index < num_outputs_; ++output_index) {
      classification_heads_.emplace_back(core::ClassificationHead{});
    }
  }

  if (num_outputs_ != classification_heads_.size()) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Got %d classifier head(s), expected %d according to "
                        "the label map.",
                        num_outputs_, classification_heads_.size()),
        TfLiteSupportStatus::kMetadataInconsistencyError);
  }

  int num_quantized_outputs = 0;
  for (int i = 0; i < num_outputs_; ++i) {
    const TfLiteTensor* output_tensor =
        TfLiteEngine::GetOutput(engine_->interpreter(), i);
    const int num_dimensions = output_tensor->dims->size;
    if (num_dimensions != 2) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat(
              "Unexpected number of dimensions for output index %d: got %dD, "
              "expected 2D (BxN with B=1).",
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
    int num_classes = output_tensor->dims->data[num_dimensions - 1];
    // If label map is not set, build a default one based on model
    // introspection. This happens if a model with partial or no metadata was
    // provided through the `model_file_with_metadata` options field.
    if (classification_heads_[i].label_map_items.empty()) {
      classification_heads_[i].label_map_items.reserve(num_classes);
      for (int class_index = 0; class_index < num_classes; ++class_index) {
        classification_heads_[i].label_map_items.emplace_back(LabelMapItem{});
      }
    }
    int num_label_map_items = classification_heads_[i].label_map_items.size();
    if (num_classes != num_label_map_items) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Got %d class(es) for output index %d, expected %d "
                          "according to the label map.",
                          output_tensor->dims->data[num_dimensions - 1], i,
                          num_label_map_items),
          TfLiteSupportStatus::kMetadataInconsistencyError);
    }
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

  if (num_quantized_outputs > 0 && num_quantized_outputs != num_outputs_) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Got %d quantized output(s), expected %d (i.e. all "
                        "provided outputs must be quantized).",
                        num_quantized_outputs, num_outputs_),
        TfLiteSupportStatus::kInvalidOutputTensorTypeError);
  }
  has_uint8_outputs_ = (num_quantized_outputs > 0);

  return absl::OkStatus();
}

tflite::support::StatusOr<ClassificationResult> AudioClassifier::Classify(
    const AudioBuffer& audio_buffer) {
  return InferWithFallback(audio_buffer);
}

// TODO(b/182537114): Extract into a common library to share between audio and
// vision tasks.
absl::Status AudioClassifier::Preprocess(
    const std::vector<TfLiteTensor*>& input_tensors,
    const AudioBuffer& audio_buffer) {
  if (input_tensors.size() != 1) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInternal, "A single input tensor is expected.");
  }
  if (audio_buffer.GetAudioFormat().channels != audio_format_.channels) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Input audio buffer channel number %d does not match "
                        "the model required audio channel number %d.",
                        audio_buffer.GetAudioFormat().channels,
                        audio_format_.channels));
  }
  if (audio_buffer.GetAudioFormat().sample_rate != audio_format_.sample_rate) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("Input audio sample rate %d does not match "
                        "the model required audio sample rate %d.",
                        audio_buffer.GetAudioFormat().sample_rate,
                        audio_format_.sample_rate));
  }
  if (audio_buffer.GetBufferSize() != input_buffer_size_) {
    return tflite::support::CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Input audio buffer size %d does not match the model required "
            "input size %d.",
            audio_buffer.GetBufferSize(), input_buffer_size_),
        TfLiteSupportStatus::kInvalidArgumentError);
  }
  tflite::task::core::PopulateTensor(audio_buffer.GetFloatBuffer(),
                                     input_buffer_size_, input_tensors[0]);
  return absl::OkStatus();
}

tflite::support::StatusOr<audio::ClassificationResult>
AudioClassifier::Postprocess(
    const std::vector<const TfLiteTensor*>& output_tensors,
    const AudioBuffer& audio_buffer) {
  if (output_tensors.size() != num_outputs_) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Expected %d output tensors, found %d", num_outputs_,
                        output_tensors.size()));
  }

  ClassificationResult result;
  std::vector<std::pair<int, float>> score_pairs;

  for (int i = 0; i < num_outputs_; ++i) {
    auto* classifications = result.add_classifications();
    classifications->set_head_index(i);

    const auto& head = classification_heads_[i];
    classifications->set_head_name(head.name);

    score_pairs.clear();
    score_pairs.reserve(head.label_map_items.size());

    const TfLiteTensor* output_tensor = output_tensors[i];
    if (has_uint8_outputs_) {
      const uint8* output_data =
          AssertAndReturnTypedTensor<uint8>(output_tensor);
      for (int j = 0; j < head.label_map_items.size(); ++j) {
        score_pairs.emplace_back(j, output_tensor->params.scale *
                                        (static_cast<int>(output_data[j]) -
                                         output_tensor->params.zero_point));
      }
    } else {
      const float* output_data =
          AssertAndReturnTypedTensor<float>(output_tensor);
      for (int j = 0; j < head.label_map_items.size(); ++j) {
        score_pairs.emplace_back(j, output_data[j]);
      }
    }

    int num_results =
        options_->max_results() >= 0
            ? std::min(static_cast<int>(head.label_map_items.size()),
                       options_->max_results())
            : head.label_map_items.size();
    float score_threshold = options_->has_score_threshold()
                                ? options_->score_threshold()
                                : head.score_threshold;

    if (class_name_set_.values.empty()) {
      // Partially sort in descending order (higher score is better).
      absl::c_partial_sort(
          score_pairs, score_pairs.begin() + num_results,
          [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.second > b.second;
          });

      for (int j = 0; j < num_results; ++j) {
        float score = score_pairs[j].second;
        if (score < score_threshold) {
          break;
        }
        auto* cl = classifications->add_classes();
        cl->set_index(score_pairs[j].first);
        cl->set_score(score);
      }
    } else {
      // Sort in descending order (higher score is better).
      absl::c_sort(score_pairs, [](const std::pair<int, float>& a,
                                   const std::pair<int, float>& b) {
        return a.second > b.second;
      });

      for (int j = 0; j < head.label_map_items.size(); ++j) {
        float score = score_pairs[j].second;
        if (score < score_threshold ||
            classifications->classes_size() >= num_results) {
          break;
        }

        const int class_index = score_pairs[j].first;
        const std::string& class_name = head.label_map_items[class_index].name;

        bool class_name_found = class_name_set_.values.contains(class_name);

        if ((!class_name_found && class_name_set_.is_allowlist) ||
            (class_name_found && !class_name_set_.is_allowlist)) {
          continue;
        }

        auto* cl = classifications->add_classes();
        cl->set_index(class_index);
        cl->set_score(score);
      }
    }
  }

  RETURN_IF_ERROR(FillResultsFromLabelMaps(&result));

  return result;
}

// TODO(b/182537114): Extract into a common library to share between audio and
// vision tasks.
absl::Status AudioClassifier::FillResultsFromLabelMaps(
    audio::ClassificationResult* result) {
  for (int i = 0; i < result->classifications_size(); ++i) {
    audio::Classifications* classifications =
        result->mutable_classifications(i);
    int head_index = classifications->head_index();
    if (head_index < 0 || head_index >= classification_heads_.size()) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument,
          absl::StrFormat("Invalid head index (%d) with respect to total "
                          "number of classification heads (%d).",
                          head_index, classification_heads_.size()),
          TfLiteSupportStatus::kMetadataInconsistencyError);
    }
    const std::vector<core::LabelMapItem>& label_map_items =
        classification_heads_[head_index].label_map_items;
    for (int j = 0; j < classifications->classes_size(); ++j) {
      Class* current_class = classifications->mutable_classes(j);
      int current_class_index = current_class->index();
      if (current_class_index < 0 ||
          current_class_index >= label_map_items.size()) {
        return CreateStatusWithPayload(
            StatusCode::kInvalidArgument,
            absl::StrFormat("Invalid class index (%d) with respect to label "
                            "map size (%d) for head #%d.",
                            current_class_index, label_map_items.size(),
                            head_index),
            TfLiteSupportStatus::kMetadataInconsistencyError);
      }
      const std::string& name = label_map_items[current_class_index].name;
      if (!name.empty()) {
        current_class->set_class_name(name);
      }
      const std::string& display_name =
          label_map_items[current_class_index].display_name;
      if (!display_name.empty()) {
        current_class->set_display_name(display_name);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace audio
}  // namespace task
}  // namespace tflite
