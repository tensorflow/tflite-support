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
#include "tensorflow_lite_support/examples/task/text/desktop/universal_sentence_encoder_qa_op_resolver.h"

#include "absl/memory/memory.h"  // from @com_google_absl
#include "tensorflow/lite/kernels/register.h"

namespace tflite {
namespace ops {
namespace custom {
TfLiteRegistration* Register_SENTENCEPIECE_TOKENIZER();
TfLiteRegistration* Register_RAGGED_TENSOR_TO_TENSOR();
}  // namespace custom
}  // namespace ops
}  // namespace tflite

namespace tflite {
namespace task {
namespace text {

// Creates custom op resolver for USE QA task.
std::unique_ptr<tflite::OpResolver> CreateQACustomOpResolver() {
  auto resolver = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  resolver->AddCustom(
      "TFSentencepieceTokenizeOp",
      ::tflite::ops::custom::Register_SENTENCEPIECE_TOKENIZER());
  resolver->AddCustom(
      "RaggedTensorToTensor",
      ::tflite::ops::custom::Register_RAGGED_TENSOR_TO_TENSOR());
  return resolver;
}

}  // namespace text
}  // namespace task
}  // namespace tflite
