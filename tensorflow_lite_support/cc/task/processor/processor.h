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
#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_PROCESSOR_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_PROCESSOR_H_

#include <initializer_list>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"

namespace tflite {
namespace task {
namespace processor {

// Abstract base class for all Preprocessors.
// Preprocessor is a helper class that converts input structured data (such as
// image) to raw bytes and populates the associated tensors in the interpreter.
//
// As a convention, child class needs to implement a factory `Create` method to
// initialize and bind tensors.
//
// Example usage:
// auto processor = MyPreprocessor::Create(
//   /* input_tensors */ {0}, engine, option);
// // Populate the associate tensors.
// processor->Preprocess(...);
class Preprocessor {
 public:
  Preprocessor() = default;
  virtual ~Preprocessor() = default;

  // Preprocessor is neither copyable nor movable.
  Preprocessor(const Preprocessor&) = delete;
  Preprocessor& operator=(const Preprocessor&) = delete;

 protected:
  explicit Preprocessor(core::TfLiteEngine* engine,
                        const std::initializer_list<int> input_indices)
      : engine_(engine), input_indices_(input_indices) {}

  core::TfLiteEngine* engine_;
  const std::vector<int> input_indices_;

  // Get the associated input tensor.
  // Note: Calling this method before `VerifyAndInit` method will cause a crash.
  // Note: Caller is responsible for passing in a valid `i`.
  inline TfLiteTensor* Tensor(int i = 0) const {
    return engine_->GetInput(engine_->interpreter(), input_indices_.at(i));
  }

  // Get the associated input metadata.
  // Note: Calling this method before `VerifyAndInit` method will cause a crash.
  // Note: Caller is responsible for passing in a valid `i`.
  inline const tflite::TensorMetadata* Metadata(int i = 0) const {
    return engine_->metadata_extractor()->GetInputTensorMetadata(
        input_indices_.at(i));
  }

  static absl::Status SanityCheck(
      int num_expected_tensors, core::TfLiteEngine* engine,
      const std::initializer_list<int> input_indices,
      bool requires_metadata = true);
};

// Abstract base class for all Postprocessors.
// Postprocessor is a helper class to convert tensor value to structured
// data.
// As a convention, child class needs to implement a factory `Create` method to
// initialize and bind tensors.
//
// Example usage:
// auto processor = MyPostprocessor::Create(
//   /* output_tensors */ {0}, engine, option);
// // Populate the associate tensors.
// auto value = processor->Postprocess();
class Postprocessor {
 public:
  Postprocessor() = default;
  virtual ~Postprocessor() = default;

  // Preprocessor is neither copyable nor movable.
  Postprocessor(const Postprocessor&) = delete;
  Postprocessor& operator=(const Postprocessor&) = delete;

 protected:
  explicit Postprocessor(core::TfLiteEngine* engine,
                         const std::initializer_list<int> output_indices)
      : engine_(engine), output_indices_(output_indices) {}

  core::TfLiteEngine* engine_;
  const std::vector<int> output_indices_;

  // Get the associated output tensor.
  // Note: Calling this method before `VerifyAndInit` method will cause a crash.
  // Note: Caller is responsible for passing in a valid `i`.
  inline TfLiteTensor* Tensor(int i = 0) const {
    return engine_->GetOutput(engine_->interpreter(), output_indices_.at(i));
  }

  // Get the associated output metadata.
  // Note: Calling this method before `VerifyAndInit` method will cause a crash.
  // Note: Caller is responsible for passing in a valid `i`.
  inline const tflite::TensorMetadata* Metadata(int i = 0) const {
    return engine_->metadata_extractor()->GetOutputTensorMetadata(
        output_indices_.at(i));
  }

  static absl::Status SanityCheck(
      int num_expected_tensors, core::TfLiteEngine* engine,
      const std::initializer_list<int> output_indices,
      bool requires_metadata = true);
};
}  // namespace processor
}  // namespace task
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_PROCESSOR_PROCESSOR_H_
