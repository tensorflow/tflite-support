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

/**
 * Sentencepiece TFLite tokenizer implementation.
 */
#include "tensorflow_lite_support/custom_ops/kernel/sentencepiece/sentencepiece_tokenizer_tflite.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow_lite_support/custom_ops/kernel/sentencepiece/optimized_encoder.h"
#include "tensorflow_lite_support/custom_ops/kernel/sentencepiece/sentencepiece_tokenizer.h"

namespace tflite {
namespace ops {
namespace custom {
namespace sentencepiece {
namespace tokenizer {

constexpr int kOutputValuesInd = 0;
constexpr int kOutputSplitsInd = 1;

namespace {

TfLiteIntArray* CreateSizeArray(const std::initializer_list<int>& sizes) {
  TfLiteIntArray* array_size = TfLiteIntArrayCreate(sizes.size());
  int index = 0;
  for (const int size : sizes) {
    array_size->data[index++] = size;
  }
  return array_size;
}

}  // namespace

TfLiteStatus Prepare(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  // TODO(andram): Enable this validation separately since it is a riskier
  // change and we should test it separately. Validate that the node has all
  // required inputs and outputs.
  // TF_LITE_OPAQUE_ENSURE(context, TfLiteOpaqueNodeNumberOfInputs(node) == 7);
  // TF_LITE_OPAQUE_ENSURE_EQ(context, TfLiteOpaqueNodeNumberOfOutputs(node),
  //                         2);

  // Both output values and splits are dynamic since their shapes depend on the
  // tokenized string lengths determined at evaluation time.
  TfLiteOpaqueTensor* output_values =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputValuesInd);
  TF_LITE_OPAQUE_ENSURE(context, output_values != nullptr);
  TfLiteOpaqueTensorSetAllocationTypeToDynamic(output_values);

  TfLiteOpaqueTensor* output_splits =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputSplitsInd);
  TF_LITE_OPAQUE_ENSURE(context, output_splits != nullptr);
  TfLiteOpaqueTensorSetAllocationTypeToDynamic(output_splits);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteOpaqueContext* context, TfLiteOpaqueNode* node) {
  // Retrieve the serialized SentencePiece model buffer.
  const TfLiteOpaqueTensor* model_tensor =
      TfLiteOpaqueNodeGetInput(context, node, tensorflow::ops::kSPModelIndex);
  TF_LITE_OPAQUE_ENSURE(context, model_tensor != nullptr);
  const void* model_buffer_data = TfLiteOpaqueTensorData(model_tensor);
  TF_LITE_OPAQUE_ENSURE(context, model_buffer_data != nullptr);

  // Retrieve the input text string tensor.
  const TfLiteOpaqueTensor* input_text =
      TfLiteOpaqueNodeGetInput(context, node, tensorflow::ops::kInputIndex);
  TF_LITE_OPAQUE_ENSURE(context, input_text != nullptr);

  // Retrieve boolean control configuration options.
  const TfLiteOpaqueTensor* add_bos_tensor =
      TfLiteOpaqueNodeGetInput(context, node, tensorflow::ops::kAddBOSInput);
  TF_LITE_OPAQUE_ENSURE(context, add_bos_tensor != nullptr);
  const bool* add_bos_data =
      static_cast<const bool*>(TfLiteOpaqueTensorData(add_bos_tensor));
  TF_LITE_OPAQUE_ENSURE(context, add_bos_data != nullptr);
  const bool add_bos = add_bos_data[0];

  const TfLiteOpaqueTensor* add_eos_tensor =
      TfLiteOpaqueNodeGetInput(context, node, tensorflow::ops::kAddEOSInput);
  TF_LITE_OPAQUE_ENSURE(context, add_eos_tensor != nullptr);
  const bool* add_eos_data =
      static_cast<const bool*>(TfLiteOpaqueTensorData(add_eos_tensor));
  TF_LITE_OPAQUE_ENSURE(context, add_eos_data != nullptr);
  const bool add_eos = add_eos_data[0];

  const TfLiteOpaqueTensor* reverse_tensor =
      TfLiteOpaqueNodeGetInput(context, node, tensorflow::ops::kReverseInput);
  TF_LITE_OPAQUE_ENSURE(context, reverse_tensor != nullptr);
  const bool* reverse_data =
      static_cast<const bool*>(TfLiteOpaqueTensorData(reverse_tensor));
  TF_LITE_OPAQUE_ENSURE(context, reverse_data != nullptr);
  const bool reverse = reverse_data[0];

  // Tokenize each string in the input tensor using the SentencePiece model.
  std::vector<int32_t> encoded;
  std::vector<int32_t> splits;
  const int num_strings = TfLiteOpaqueTensorGetStringCount(input_text);
  splits.reserve(num_strings);
  for (int i = 0; i < num_strings; ++i) {
    const char* str = nullptr;
    int len = 0;
    TF_LITE_OPAQUE_ENSURE_EQ(
        context, TfLiteOpaqueTensorGetString(input_text, i, &str, &len),
        kTfLiteOk);
    const auto res = EncodeString(std::string(str, len), model_buffer_data,
                                  add_bos, add_eos, reverse);
    TF_LITE_OPAQUE_ENSURE_MSG(context, res.type == EncoderResultType::SUCCESS,
                              "Sentencepiece conversion failed");
    std::copy(res.codes.begin(), res.codes.end(), std::back_inserter(encoded));
    splits.emplace_back(encoded.size());
  }

  // Resize and populate output values tensor.
  TfLiteOpaqueTensor* output_values =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputValuesInd);
  TF_LITE_OPAQUE_ENSURE(context, output_values != nullptr);
  TF_LITE_OPAQUE_ENSURE_EQ(
      context,
      TfLiteOpaqueContextResizeTensor(
          context, output_values,
          CreateSizeArray({static_cast<int>(encoded.size())})),
      kTfLiteOk);
  if (!encoded.empty()) {
    int32_t* output_values_flat =
        static_cast<int32_t*>(TfLiteOpaqueTensorData(output_values));
    TF_LITE_OPAQUE_ENSURE(context, output_values_flat != nullptr);
    std::copy(encoded.begin(), encoded.end(), output_values_flat);
  }

  // Resize and populate output splits tensor.
  TfLiteOpaqueTensor* output_splits =
      TfLiteOpaqueNodeGetOutput(context, node, kOutputSplitsInd);
  TF_LITE_OPAQUE_ENSURE(context, output_splits != nullptr);
  TF_LITE_OPAQUE_ENSURE_EQ(
      context,
      TfLiteOpaqueContextResizeTensor(
          context, output_splits,
          CreateSizeArray({static_cast<int>(splits.size() + 1)})),
      kTfLiteOk);
  int32_t* output_splits_flat =
      static_cast<int32_t*>(TfLiteOpaqueTensorData(output_splits));
  TF_LITE_OPAQUE_ENSURE(context, output_splits_flat != nullptr);
  *output_splits_flat = 0;
  std::copy(splits.begin(), splits.end(), output_splits_flat + 1);
  return kTfLiteOk;
}

}  // namespace tokenizer
}  // namespace sentencepiece

TfLiteOperator* RegisterExternal_SENTENCEPIECE_TOKENIZER() {
  static TfLiteOperator* r = []() {
    auto* op =
        TfLiteOperatorCreate(kTfLiteBuiltinCustom, "TFSentencepieceTokenizeOp",
                             /*version=*/1, /*user_data=*/nullptr);
    TfLiteOperatorSetPrepare(op, sentencepiece::tokenizer::Prepare);
    TfLiteOperatorSetInvoke(op, sentencepiece::tokenizer::Eval);
    return op;
  }();
  return r;
}

TfLiteRegistration* Register_SENTENCEPIECE_TOKENIZER() {
  static TfLiteOperator* reg_external =
      RegisterExternal_SENTENCEPIECE_TOKENIZER();
  static TfLiteRegistration reg = {
      /*.init=*/nullptr,
      /*.free=*/nullptr,
      /*.prepare=*/nullptr,
      /*.invoke=*/nullptr,
      /*.profiling_string=*/nullptr,
      /*.builtin_code=*/kTfLiteBuiltinCustom,
      /*.custom_name=*/"TFSentencepieceTokenizeOp",
      /*.version=*/1,
      /*.registration_external=*/reg_external,
  };
  return &reg;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
