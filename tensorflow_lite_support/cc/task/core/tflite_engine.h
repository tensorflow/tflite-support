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

#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_CORE_TFLITE_ENGINE_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_CORE_TFLITE_ENGINE_H_

#include <sys/mman.h>

#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow_lite_support/cc/port/tflite_wrapper.h"
#include "tensorflow_lite_support/cc/task/core/external_file_handler.h"
#include "tensorflow_lite_support/cc/task/core/proto/external_file_proto_inc.h"
#include "tensorflow_lite_support/metadata/cc/metadata_extractor.h"

namespace tflite {
namespace task {
namespace core {

// TfLiteEngine encapsulates logic for TFLite model initialization, inference
// and error reporting.
class TfLiteEngine {
 public:
  explicit TfLiteEngine(
      std::unique_ptr<tflite::IterableOpResolver> resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());
  // Model is neither copyable nor movable.
  TfLiteEngine(const TfLiteEngine&) = delete;
  TfLiteEngine& operator=(const TfLiteEngine&) = delete;

  // Accessors.
  tflite::FlatBufferModel* model() const { return model_.get(); }
  tflite::Interpreter* interpreter() const { return interpreter_.get(); }
  tflite::support::TfLiteInterpreterWrapper* interpreter_wrapper() {
    return &interpreter_;
  }
  const tflite::metadata::ModelMetadataExtractor* metadata_extractor() const {
    return model_metadata_extractor_.get();
  }

  // Builds the TF Lite FlatBufferModel (model_) from the raw FlatBuffer data
  // whose ownership remains with the caller, and which must outlive the current
  // object. This performs extra verification on the input data using
  // tflite::Verify.
  absl::Status BuildModelFromFlatBuffer(const char* buffer_data,
                                        size_t buffer_size);

  // Builds the TF Lite model from a given file.
  absl::Status BuildModelFromFile(const std::string& file_name);

  // Builds the TF Lite model from a given file descriptor using mmap(2).
  absl::Status BuildModelFromFileDescriptor(int file_descriptor);

  // Builds the TFLite model from the provided ExternalFile proto, which must
  // outlive the current object.
  absl::Status BuildModelFromExternalFileProto(
      const ExternalFile* external_file);

  // Initializes interpreter with encapsulated model.
  // Note: setting num_threads to -1 has for effect to let TFLite runtime set
  // the value.
  absl::Status InitInterpreter(int num_threads = 1);

  // Same as above, but allows specifying `compute_settings` for acceleration.
  absl::Status InitInterpreter(
      const tflite::proto::ComputeSettings& compute_settings,
      int num_threads = 1);

  // Cancels the on-going `Invoke()` call if any and if possible. This method
  // can be called from a different thread than the one where `Invoke()` is
  // running.
  void Cancel() { interpreter_.Cancel(); }

 protected:
  // TF Lite's DefaultErrorReporter() outputs to stderr. This one captures the
  // error into a string so that it can be used to complement tensorflow::Status
  // error messages.
  struct ErrorReporter : public tflite::ErrorReporter {
    // Last error message captured by this error reporter.
    char error_message[256];
    int Report(const char* format, va_list args) override;
  };
  // Custom error reporter capturing low-level TF Lite error messages.
  ErrorReporter error_reporter_;

 private:
  // Direct wrapper around tflite::TfLiteVerifier which checks the integrity of
  // the FlatBuffer data provided as input.
  class Verifier : public tflite::TfLiteVerifier {
   public:
    explicit Verifier(const tflite::OpResolver* op_resolver)
        : op_resolver_(op_resolver) {}
    bool Verify(const char* data, int length,
                tflite::ErrorReporter* reporter) override;
    // The OpResolver to be used to build the TF Lite interpreter.
    const tflite::OpResolver* op_resolver_;
  };

  absl::Status InitializeFromModelFileHandler();

  // TF Lite model and interpreter for actual inference.
  std::unique_ptr<tflite::FlatBufferModel> model_;

  // Interpreter wrapper built from the model.
  tflite::support::TfLiteInterpreterWrapper interpreter_;

  // TFLite Metadata extractor built from the model.
  std::unique_ptr<tflite::metadata::ModelMetadataExtractor>
      model_metadata_extractor_;

  // Mechanism used by TF Lite to map Ops referenced in the FlatBuffer model to
  // actual implementation. Defaults to TF Lite BuiltinOpResolver.
  std::unique_ptr<tflite::OpResolver> resolver_;

  // Extra verifier for FlatBuffer input data.
  Verifier verifier_;

  // ExternalFile and corresponding ExternalFileHandler for models loaded from
  // disk or file descriptor.
  ExternalFile external_file_;
  std::unique_ptr<ExternalFileHandler> model_file_handler_;
};

}  // namespace core
}  // namespace task
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_CORE_TFLITE_ENGINE_H_
