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

#include "tensorflow_lite_support/cc/text/tokenizers/tokenizer_utils.h"

#include "absl/status/status.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h"
#include "tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

namespace tflite::support::text::tokenizer {

using ::tflite::ProcessUnit;
using ::tflite::SentencePieceTokenizerOptions;
using ::tflite::support::CreateStatusWithPayload;

StatusOr<std::unique_ptr<Tokenizer>> CreateTokenizerFromMetadata(
    const tflite::metadata::ModelMetadataExtractor& metadata_extractor) {
  const ProcessUnit* tokenizer_process_unit =
      metadata_extractor.GetInputProcessUnit(kTokenizerProcessUnitIndex);
  if (tokenizer_process_unit == nullptr) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        "No input process unit found from metadata.",
        TfLiteSupportStatus::kMetadataInvalidTokenizerError);
  }
  if (tokenizer_process_unit->options_type() ==
      ProcessUnitOptions_BertTokenizerOptions) {
    const tflite::BertTokenizerOptions* options =
        tokenizer_process_unit->options_as<tflite::BertTokenizerOptions>();
    absl::string_view vocab_buffer;
    if (options->vocab_file() == nullptr || options->vocab_file()->size() < 1 ||
        options->vocab_file()->Get(0)->name() == nullptr) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Invalid vocab_file from input process unit.",
          TfLiteSupportStatus::kMetadataInvalidTokenizerError);
    }
    ASSIGN_OR_RETURN(vocab_buffer,
                     metadata_extractor.GetAssociatedFile(
                         options->vocab_file()->Get(0)->name()->str()));
    return absl::make_unique<BertTokenizer>(vocab_buffer.data(),
                                            vocab_buffer.size());
  }

  if (tokenizer_process_unit->options_type() ==
      ProcessUnitOptions_SentencePieceTokenizerOptions) {
    const SentencePieceTokenizerOptions* options =
        tokenizer_process_unit->options_as<SentencePieceTokenizerOptions>();
    absl::string_view model_buffer;
    if (options->sentencePiece_model() == nullptr ||
        options->sentencePiece_model()->size() < 1 ||
        options->sentencePiece_model()->Get(0)->name() == nullptr) {
      return CreateStatusWithPayload(
          absl::StatusCode::kInvalidArgument,
          "Invalid sentencePiece_model from input process unit.",
          TfLiteSupportStatus::kMetadataInvalidTokenizerError);
    }
    ASSIGN_OR_RETURN(
        model_buffer,
        metadata_extractor.GetAssociatedFile(
            options->sentencePiece_model()->Get(0)->name()->str()));
    // TODO(b/160647204): Extract sentence piece model vocabulary
    return absl::make_unique<SentencePieceTokenizer>(model_buffer.data(),
                                                     model_buffer.size());
  } else {
    return CreateStatusWithPayload(
        absl::StatusCode::kNotFound,
        absl::StrCat("Incorrect options_type:",
                     tokenizer_process_unit->options_type()),
        TfLiteSupportStatus::kMetadataInvalidTokenizerError);
  }
}

}  // namespace tflite::support::text::tokenizer
