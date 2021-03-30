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

#include "tensorflow_lite_support/cc/port/default/tflite_wrapper.h"

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/interpreter_utils.h"
#include "tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"

namespace tflite {
namespace support {

namespace {
using tflite::delegates::DelegatePluginRegistry;
using tflite::delegates::InterpreterUtils;
using tflite::proto::ComputeSettings;
using tflite::proto::Delegate;
}  // namespace

/* static */
absl::Status TfLiteInterpreterWrapper::SanityCheckComputeSettings(
    const ComputeSettings& compute_settings) {
  Delegate delegate = compute_settings.tflite_settings().delegate();
  if (delegate != Delegate::NONE && delegate != Delegate::GPU &&
      delegate != Delegate::HEXAGON && delegate != Delegate::NNAPI &&
      delegate != Delegate::XNNPACK && delegate != Delegate::EDGETPU_CORAL) {
    return absl::UnimplementedError(absl::StrFormat(
        "Using delegate '%s' is not supported.", Delegate_Name(delegate)));
  }
  return absl::OkStatus();
}

TfLiteInterpreterWrapper::TfLiteInterpreterWrapper()
    : delegate_(nullptr, nullptr), got_error_do_not_delegate_anymore_(false) {}

absl::Status TfLiteInterpreterWrapper::InitializeWithFallback(
    std::function<absl::Status(std::unique_ptr<tflite::Interpreter>*)>
        interpreter_initializer,
    const ComputeSettings& compute_settings) {
  return InitializeWithFallback(
      [interpreter_initializer](
          const InterpreterCreationResources& /*resources*/,
          std::unique_ptr<tflite::Interpreter>* interpreter_out)
          -> absl::Status { return interpreter_initializer(interpreter_out); },
      compute_settings);
}

absl::Status TfLiteInterpreterWrapper::InitializeWithFallback(
    std::function<absl::Status(const InterpreterCreationResources&,
                               std::unique_ptr<tflite::Interpreter>*)>
        interpreter_initializer,
    const ComputeSettings& compute_settings) {
  // Store interpreter initializer if not already here.
  if (interpreter_initializer_) {
    return absl::FailedPreconditionError(
        "InitializeWithFallback already called.");
  }
  interpreter_initializer_ = std::move(interpreter_initializer);

  // Sanity check and copy ComputeSettings.
  RETURN_IF_ERROR(SanityCheckComputeSettings(compute_settings));
  compute_settings_ = compute_settings;

  // Initialize fallback behavior.
  fallback_on_compilation_error_ =
      compute_settings_.tflite_settings()
          .fallback_settings()
          .allow_automatic_fallback_on_compilation_error() ||
      // Deprecated, keep supporting for backward compatibility.
      compute_settings_.tflite_settings()
          .nnapi_settings()
          .fallback_settings()
          .allow_automatic_fallback_on_compilation_error();
  fallback_on_execution_error_ =
      compute_settings_.tflite_settings()
          .fallback_settings()
          .allow_automatic_fallback_on_execution_error() ||
      // Deprecated, keep supporting for backward compatibility.
      compute_settings_.tflite_settings()
          .nnapi_settings()
          .fallback_settings()
          .allow_automatic_fallback_on_execution_error();

  return InitializeWithFallbackAndResize();
}

absl::Status TfLiteInterpreterWrapper::AllocateTensors() {
  if (interpreter_->AllocateTensors() != kTfLiteOk) {
    return absl::InternalError("AllocateTensors() failed.");
  }
  return absl::OkStatus();
}

// TODO(b/173406463): the `resize` parameter is going to be used by
// ResizeAndAllocateTensors functions, coming soon.
absl::Status TfLiteInterpreterWrapper::InitializeWithFallbackAndResize(
    std::function<absl::Status(Interpreter* interpreter)> resize) {
  RETURN_IF_ERROR(
      interpreter_initializer_(InterpreterCreationResources(), &interpreter_));
  RETURN_IF_ERROR(resize(interpreter_.get()));
  if (compute_settings_.tflite_settings().cpu_settings().num_threads() != -1) {
    if (interpreter_->SetNumThreads(
            compute_settings_.tflite_settings().cpu_settings().num_threads()) !=
        kTfLiteOk) {
      return absl::InternalError("Failed setting number of CPU threads");
    }
  }
  SetTfLiteCancellation();

  if (got_error_do_not_delegate_anymore_ ||
      compute_settings_.tflite_settings().delegate() == Delegate::NONE) {
    // Just allocate tensors and return.
    delegate_.reset(nullptr);
    return AllocateTensors();
  }

  // Initialize delegate and modify graph.
  RETURN_IF_ERROR(InitializeDelegate());
  if (interpreter_->ModifyGraphWithDelegate(delegate_.get()) != kTfLiteOk) {
    // If a compilation error occurs, stop delegation from happening in the
    // future.
    got_error_do_not_delegate_anymore_ = true;
    delegate_.reset(nullptr);
    if (!fallback_on_compilation_error_) {
      // If instructed not to fallback, return error.
      return absl::InternalError(absl::StrFormat(
          "ModifyGraphWithDelegate() failed for delegate '%s'.",
          Delegate_Name(compute_settings_.tflite_settings().delegate())));
    }
  }

  // The call to ModifyGraphWithDelegate() leaves the interpreter in a usable
  // state in case of failure: calling AllocateTensors() will silently fallback
  // on CPU in such a situation.
  return AllocateTensors();
}

absl::Status TfLiteInterpreterWrapper::InitializeDelegate() {
  if (delegate_ == nullptr) {
    Delegate which_delegate = compute_settings_.tflite_settings().delegate();
    const tflite::ComputeSettings* compute_settings =
        tflite::ConvertFromProto(compute_settings_, &flatbuffers_builder_);

    if (which_delegate == Delegate::NNAPI) {
      RETURN_IF_ERROR(
          LoadDelegatePlugin("Nnapi", *compute_settings->tflite_settings()));
    } else if (which_delegate == Delegate::HEXAGON) {
      RETURN_IF_ERROR(
          LoadDelegatePlugin("Hexagon", *compute_settings->tflite_settings()));
    } else if (which_delegate == Delegate::GPU) {
      RETURN_IF_ERROR(
          LoadDelegatePlugin("Gpu", *compute_settings->tflite_settings()));
    } else if (which_delegate == Delegate::EDGETPU) {
      RETURN_IF_ERROR(
          LoadDelegatePlugin("EdgeTpu", *compute_settings->tflite_settings()));
    } else if (which_delegate == Delegate::EDGETPU_CORAL) {
      RETURN_IF_ERROR(LoadDelegatePlugin("EdgeTpuCoral",
                                         *compute_settings->tflite_settings()));
    } else if (which_delegate == Delegate::XNNPACK) {
      RETURN_IF_ERROR(
          LoadDelegatePlugin("XNNPack", *compute_settings->tflite_settings()));
    }
  }
  return absl::OkStatus();
}

absl::Status TfLiteInterpreterWrapper::InvokeWithFallback(
    const std::function<absl::Status(tflite::Interpreter* interpreter)>&
        set_inputs) {
  RETURN_IF_ERROR(set_inputs(interpreter_.get()));
  // Reset cancel flag before calling `Invoke()`.
  cancel_flag_.Set(false);
  TfLiteStatus status = kTfLiteError;
  if (fallback_on_execution_error_) {
    status = InterpreterUtils::InvokeWithCPUFallback(interpreter_.get());
  } else {
    status = interpreter_->Invoke();
  }
  if (status == kTfLiteOk) {
    return absl::OkStatus();
  }
  // Assume InvokeWithoutFallback() is guarded under caller's synchronization.
  // Assume the inference is cancelled successfully if Invoke() returns
  // kTfLiteError and the cancel flag is `true`.
  if (status == kTfLiteError && cancel_flag_.Get()) {
    return absl::CancelledError("Invoke() cancelled.");
  }
  if (delegate_) {
    // Mark that an error occurred so that later invocations immediately
    // fallback to CPU.
    got_error_do_not_delegate_anymore_ = true;
    // InvokeWithCPUFallback returns `kTfLiteDelegateError` in case of
    // *successful* fallback: convert it to an OK status.
    if (status == kTfLiteDelegateError) {
      return absl::OkStatus();
    }
  }
  return absl::InternalError("Invoke() failed.");
}

absl::Status TfLiteInterpreterWrapper::InvokeWithoutFallback() {
  // Reset cancel flag before calling `Invoke()`.
  cancel_flag_.Set(false);
  TfLiteStatus status = interpreter_->Invoke();
  if (status != kTfLiteOk) {
    // Assume InvokeWithoutFallback() is guarded under caller's synchronization.
    // Assume the inference is cancelled successfully if Invoke() returns
    // kTfLiteError and the cancel flag is `true`.
    if (status == kTfLiteError && cancel_flag_.Get()) {
      return absl::CancelledError("Invoke() cancelled.");
    }
    return absl::InternalError("Invoke() failed.");
  }
  return absl::OkStatus();
}

void TfLiteInterpreterWrapper::Cancel() { cancel_flag_.Set(true); }

void TfLiteInterpreterWrapper::SetTfLiteCancellation() {
  // Create a cancellation check function and set to the TFLite interpreter.
  auto check_cancel_flag = [](void* data) {
    auto* cancel_flag = reinterpret_cast<CancelFlag*>(data);
    return cancel_flag->Get();
  };
  interpreter_->SetCancellationFunction(reinterpret_cast<void*>(&cancel_flag_),
                                        check_cancel_flag);
}

absl::Status TfLiteInterpreterWrapper::LoadDelegatePlugin(
    const std::string& name, const tflite::TFLiteSettings& tflite_settings) {
  delegate_plugin_ = DelegatePluginRegistry::CreateByName(
      absl::StrFormat("%sPlugin", name), tflite_settings);

  if (delegate_plugin_ == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Could not create %s plugin. Have you linked in the %s_plugin target?",
        name, name));
  }

  delegate_ = delegate_plugin_->Create();
  if (delegate_ == nullptr) {
    return absl::InternalError(
        absl::StrFormat("Plugin did not create %s delegate.", name));
  }

  return absl::OkStatus();
}

}  // namespace support
}  // namespace tflite
