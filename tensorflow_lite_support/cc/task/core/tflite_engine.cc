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

#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"

#include <unistd.h>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/lite/tools/verifier.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/core/external_file_handler.h"

namespace tflite {
namespace task {
namespace core {

#ifdef __ANDROID__
// https://github.com/opencv/opencv/issues/14906
// "ios_base::Init" object is not a part of Android's "iostream" header (in case
// of clang toolchain, NDK 20).
//
// Ref1:
// https://en.cppreference.com/w/cpp/io/ios_base/Init
//       The header <iostream> behaves as if it defines (directly or indirectly)
//       an instance of std::ios_base::Init with static storage duration
//
// Ref2:
// https://github.com/gcc-mirror/gcc/blob/gcc-8-branch/libstdc%2B%2B-v3/include/std/iostream#L73-L74
static std::ios_base::Init s_iostream_initializer;
#endif

namespace {
using ::absl::StatusCode;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::TfLiteSupportStatus;
}  // namespace

TfLiteEngine::TfLiteEngine(std::unique_ptr<tflite::OpResolver> resolver)
    : resolver_(std::move(resolver)), verifier_(resolver_.get()) {}

int TfLiteEngine::ErrorReporter::Report(const char* format, va_list args) {
  return std::vsnprintf(error_message, sizeof(error_message), format, args);
}

bool TfLiteEngine::Verifier::Verify(const char* data, int length,
                                    tflite::ErrorReporter* reporter) {
  return tflite::Verify(data, length, *op_resolver_, reporter);
}

absl::Status TfLiteEngine::InitializeFromModelFileHandler() {
  const char* buffer_data = model_file_handler_->GetFileContent().data();
  size_t buffer_size = model_file_handler_->GetFileContent().size();
  model_ = tflite::FlatBufferModel::VerifyAndBuildFromBuffer(
      buffer_data, buffer_size, &verifier_, &error_reporter_);

  if (model_ == nullptr) {
    // To be replaced with a proper switch-case when TF Lite model builder
    // returns a `TfLiteStatus` code capturing this type of error.
    if (absl::StrContains(error_reporter_.error_message,
                          "The model is not a valid Flatbuffer")) {
      return CreateStatusWithPayload(
          StatusCode::kInvalidArgument, error_reporter_.error_message,
          TfLiteSupportStatus::kInvalidFlatBufferError);
    } else {
      // TODO(b/154917059): augment status with another `TfLiteStatus` code when
      // ready. And use a new `TfLiteStatus::kCoreTfLiteError` for the TFLS
      // code, instead of the unspecified `kError`.
      return CreateStatusWithPayload(
          StatusCode::kUnknown,
          absl::StrCat(
              "Could not build model from the provided pre-loaded flatbuffer: ",
              error_reporter_.error_message));
    }
  }

  ASSIGN_OR_RETURN(
      model_metadata_extractor_,
      tflite::metadata::ModelMetadataExtractor::CreateFromModelBuffer(
          buffer_data, buffer_size));

  return absl::OkStatus();
}

absl::Status TfLiteEngine::BuildModelFromFlatBuffer(const char* buffer_data,
                                                    size_t buffer_size) {
  if (model_) {
    return CreateStatusWithPayload(StatusCode::kInternal,
                                   "Model already built");
  }
  external_file_.set_file_content(std::string(buffer_data, buffer_size));
  ASSIGN_OR_RETURN(
      model_file_handler_,
      ExternalFileHandler::CreateFromExternalFile(&external_file_));
  return InitializeFromModelFileHandler();
}

absl::Status TfLiteEngine::BuildModelFromFile(const std::string& file_name) {
  if (model_) {
    return CreateStatusWithPayload(StatusCode::kInternal,
                                   "Model already built");
  }
  external_file_.set_file_name(file_name);
  ASSIGN_OR_RETURN(
      model_file_handler_,
      ExternalFileHandler::CreateFromExternalFile(&external_file_));
  return InitializeFromModelFileHandler();
}

absl::Status TfLiteEngine::BuildModelFromFileDescriptor(int file_descriptor) {
  if (model_) {
    return CreateStatusWithPayload(StatusCode::kInternal,
                                   "Model already built");
  }
  external_file_.mutable_file_descriptor_meta()->set_fd(file_descriptor);
  ASSIGN_OR_RETURN(
      model_file_handler_,
      ExternalFileHandler::CreateFromExternalFile(&external_file_));
  return InitializeFromModelFileHandler();
}

absl::Status TfLiteEngine::BuildModelFromExternalFileProto(
    const ExternalFile* external_file) {
  if (model_) {
    return CreateStatusWithPayload(StatusCode::kInternal,
                                   "Model already built");
  }
  ASSIGN_OR_RETURN(model_file_handler_,
                   ExternalFileHandler::CreateFromExternalFile(external_file));
  return InitializeFromModelFileHandler();
}

absl::Status TfLiteEngine::InitInterpreter(int num_threads) {
  tflite::proto::ComputeSettings compute_settings;
  return InitInterpreter(compute_settings, num_threads);
}

absl::Status TfLiteEngine::InitInterpreter(
    const tflite::proto::ComputeSettings& compute_settings, int num_threads) {
  if (model_ == nullptr) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        "TF Lite FlatBufferModel is null. Please make sure to call "
        "one of the BuildModelFrom methods before.");
  }

  auto initializer =
      [this, num_threads](std::unique_ptr<tflite::Interpreter>* interpreter_out)
      -> absl::Status {
    if (tflite::InterpreterBuilder(*model_, *resolver_)(
            interpreter_out, num_threads) != kTfLiteOk) {
      return CreateStatusWithPayload(
          StatusCode::kUnknown,
          absl::StrCat("Could not build the TF Lite interpreter: ",
                       error_reporter_.error_message));
    }
    if (interpreter_out == nullptr) {
      return CreateStatusWithPayload(StatusCode::kInternal,
                                     "TF Lite interpreter is null.");
    }
    return absl::OkStatus();
  };

  absl::Status status =
      interpreter_.InitializeWithFallback(initializer, compute_settings);

  if (!status.ok() &&
      !status.GetPayload(tflite::support::kTfLiteSupportPayload).has_value()) {
    status = CreateStatusWithPayload(status.code(), status.message());
  }
  return status;
}

}  // namespace core
}  // namespace task
}  // namespace tflite
