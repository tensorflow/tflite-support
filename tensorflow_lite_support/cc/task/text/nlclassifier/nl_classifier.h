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

#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_TEXT_NLCLASSIFIER_NL_CLASSIFIER_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_TEXT_NLCLASSIFIER_NL_CLASSIFIER_H_

#include "absl/status/status.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/base_task_api.h"
#include "tensorflow_lite_support/cc/task/core/category.h"

namespace tflite {
namespace support {
namespace task {
namespace text {
namespace nlclassifier {

// Options to identify input and output tensors of the model
struct NLClassifierOptions {
  int input_tensor_index = 0;
  int output_score_tensor_index = 0;
  // By default there is no output label tensor. The label file can be attached
  // to the output score tensor metadata. See NLClassifier for more
  // information.
  int output_label_tensor_index = -1;
  std::string input_tensor_name = "INPUT";
  std::string output_score_tensor_name = "OUTPUT_SCORE";
  std::string output_label_tensor_name = "OUTPUT_LABEL";
};

// Classifier API for NLClassification tasks, categorizes string into different
// classes.
//
// The API expects a TFLite model with the following input/output tensor:
// Input tensor:
//   (kTfLiteString) - input of the model, accepts a string.
// Output score tensor:
//   (kTfLiteUInt8/kTfLiteInt8/kTfLiteInt16/kTfLiteFloat32/kTfLiteFloat64)
//    - output scores for each class, if type is one of the Int types,
//      dequantize it to double
//    - can have an optional associated file in metadata for labels, the file
//      should be a plain text file with one label per line, the number of
//      labels should match the number of categories the model outputs.
// Output label tensor: optional
//   (kTfLiteString)
//    - output classname for each class, should be of the same length with
//      scores. If this tensor is not present, the API uses score indices as
//      classnames.
//    - will be ignored if output score tensor already has an associated label
//      file.
//
// By default the API tries to find the input/output tensors with default
// configurations in NLClassifierOptions, with tensor name prioritized over
// tensor index. The option is configurable for different TFLite models.
class NLClassifier : public core::BaseTaskApi<std::vector<core::Category>,
                                              const std::string&> {
 public:
  using BaseTaskApi::BaseTaskApi;

  // Creates a NLClassifier from TFLite model buffer.
  static StatusOr<std::unique_ptr<NLClassifier>> CreateNLClassifier(
      const char* model_buffer_data, size_t model_buffer_size,
      const NLClassifierOptions& options = {},
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());

  // Creates a NLClassifier from TFLite model file.
  static StatusOr<std::unique_ptr<NLClassifier>> CreateNLClassifier(
      const std::string& path_to_model, const NLClassifierOptions& options = {},
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());

  // Creates a NLClassifier from TFLite model file descriptor.
  static StatusOr<std::unique_ptr<NLClassifier>> CreateNLClassifier(
      int fd, const NLClassifierOptions& options = {},
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());

  // Performs classification on a string input, returns classified results.
  std::vector<core::Category> Classify(const std::string& text);

 protected:
  const NLClassifierOptions& GetOptions() const;
  void SetOptions(const NLClassifierOptions& options);
  void SetLabelsVector(std::unique_ptr<std::vector<std::string>> labels_vector);
  absl::Status Preprocess(const std::vector<TfLiteTensor*>& input_tensors,
                          const std::string& input) override;

  StatusOr<std::vector<core::Category>> Postprocess(
      const std::vector<const TfLiteTensor*>& output_tensors,
      const std::string& input) override;

  // Gets the tensor from a vector of tensors by checking tensor name first and
  // tensor index second, return nullptr if no tensor is found.
  template <typename TensorType>
  static TensorType* FindTensorWithNameOrIndex(
      const std::vector<TensorType*>& tensors,
      const flatbuffers::Vector<flatbuffers::Offset<TensorMetadata>>*
          metadata_array,
      const std::string& name, int index)  {
    if (metadata_array != nullptr && metadata_array->size() == tensors.size()) {
      for (int i = 0; i < metadata_array->size(); i++) {
        if (strcmp(name.data(), metadata_array->Get(i)->name()->c_str()) == 0) {
          return tensors[i];
        }
      }
    }

    for (TensorType* tensor : tensors) {
      if (tensor->name == name) {
        return tensor;
      }
    }
    return index >= 0 && index < tensors.size() ? tensors[index] : nullptr;
  }

  // Set options and validate model with options.
  static absl::Status CheckStatusAndSetOptions(
      const NLClassifierOptions& options, NLClassifier* nl_classifier);

 private:
  NLClassifierOptions options_;
  // labels vector initialized from output tensor's associated file, if one
  // exists.
  std::unique_ptr<std::vector<std::string>> labels_vector_;
};

}  // namespace nlclassifier
}  // namespace text
}  // namespace task
}  // namespace support
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_TEXT_NLCLASSIFIER_NL_CLASSIFIER_H_
