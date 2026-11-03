/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_lite_support/custom_ops/kernel/sentencepiece/sentencepiece_tokenizer_tflite.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/custom_ops/kernel/sentencepiece/model_converter.h"

namespace tflite {
namespace ops {
namespace custom {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::Lt;
using ::testing::SizeIs;
using ::tflite::task::JoinPath;

// Location of the SentencePiece model proto, relative to the workspace root in
// the test runfiles (see `TestWorkspaceDir()`).
constexpr char kModelPath[] =
    "third_party/tensorflow_lite_support/custom_ops/kernel/"
    "sentencepiece/testdata/sentencepiece.model";

// Golden token IDs produced by `testdata/sentencepiece.model` (no BOS/EOS, not
// reversed). Pinning exact IDs catches semantic regressions in the kernel
// (e.g. reading the wrong flag input or the wrong string) that structural
// checks on the output shapes alone would miss.
constexpr int32_t kHelloWorldIds[] = {90, 547, 58, 359, 2294};
constexpr int32_t kWorldIds[] = {359};

// BOS and EOS IDs of `testdata/sentencepiece.model`, emitted around the tokens
// when the `add_bos` / `add_eos` flags are set.
constexpr int32_t kBosId = 1;
constexpr int32_t kEosId = 2;

// Returns the root of this workspace inside the test runfiles. Runfiles are
// laid out as `<SrcDir()>/<workspace name>/<path>`, and the test runner
// exports the workspace name in `TEST_WORKSPACE`. Reading it from the
// environment, rather than hardcoding it, keeps the path valid in every build
// this file is exported to, where the workspace name differs.
std::string TestWorkspaceDir() {
  const char* workspace = std::getenv("TEST_WORKSPACE");
  return JoinPath(::testing::SrcDir(), workspace != nullptr ? workspace : "");
}

// Reads the SentencePiece model proto from the test data and converts it into
// the FlatBuffer format consumed by the kernel. Records a fatal failure and
// leaves `converted_model` untouched if the file cannot be read, so the
// converter is never run on an empty proto.
void LoadConvertedModel(std::string* converted_model) {
  const std::string model_path = JoinPath(TestWorkspaceDir(), kModelPath);
  std::ifstream model_file(model_path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(model_file.is_open()) << "Failed to open " << model_path;
  std::ostringstream model_proto;
  model_proto << model_file.rdbuf();
  *converted_model =
      sentencepiece::ConvertSentencepieceModel(model_proto.str());
  ASSERT_FALSE(converted_model->empty()) << "Model conversion failed";
}

// Test fixture that initializes the TFLite C API shims and loads/converts the
// SentencePiece model once for the whole suite, since conversion is the most
// expensive part of each test and the converted model is read-only.
class SentencepieceTokenizerOp : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_EQ(TfLiteInitializeShimsForTest(), 0);
    std::string converted_model;
    LoadConvertedModel(&converted_model);
    // The helper's ASSERTs only return from the helper, so stop here
    // explicitly to leave `converted_model_` null on failure.
    if (HasFatalFailure()) return;
    converted_model_ = new std::string(std::move(converted_model));
  }

  static void TearDownTestSuite() {
    delete converted_model_;
    converted_model_ = nullptr;
  }

  // A fatal failure in `SetUpTestSuite()` does not stop the tests from
  // running, so each test checks that the model was loaded before using it.
  void SetUp() override {
    ASSERT_NE(converted_model_, nullptr)
        << "SentencePiece model failed to load in SetUpTestSuite()";
  }

  // Returns the converted model shared by all tests in the suite.
  // Precondition: `SetUp()` succeeded.
  static const std::string& model() { return *converted_model_; }

 private:
  static inline const std::string* converted_model_ = nullptr;
};

// Serializes `strings` into the raw buffer layout TFLite uses for STRING
// tensors, so the text can be embedded as a constant tensor in the model:
//   int32 num_strings
//   int32 offsets[num_strings + 1]  (byte offsets from the buffer start; the
//                                    last entry is the total buffer size)
//   char  data[]                    (all strings concatenated, no separators)
// Embedding the strings this way avoids writing a runtime input tensor, which
// would require treating a `TfLiteTensor*` from the interpreter API as a
// `TfLiteOpaqueTensor*` -- a cast the public C API does not guarantee is valid.
std::vector<uint8_t> SerializeStringTensorBuffer(
    const std::vector<std::string>& strings) {
  // Build the header: the string count followed by each string's start offset
  // and a trailing end offset.
  const size_t header_size = sizeof(int32_t) * (strings.size() + 2);
  std::vector<int32_t> header;
  header.reserve(strings.size() + 2);
  header.push_back(static_cast<int32_t>(strings.size()));
  size_t offset = header_size;
  for (const std::string& s : strings) {
    header.push_back(static_cast<int32_t>(offset));
    offset += s.size();
  }
  header.push_back(static_cast<int32_t>(offset));

  // Copy the header bytes followed by the concatenated string bytes. memcpy is
  // used for the header to avoid type-punning the byte buffer as int32_t.
  std::vector<uint8_t> buffer(offset);
  std::memcpy(buffer.data(), header.data(), header_size);
  uint8_t* string_data = buffer.data() + header_size;
  for (const std::string& s : strings) {
    string_data = std::copy(s.begin(), s.end(), string_data);
  }
  return buffer;
}

// Test model harness that builds an in-memory TFLite FlatBuffer with
// TFSentencepieceTokenizeOp and executes it using the TFLite C API.
class SentencepieceTokenizerOpModel {
 public:
  SentencepieceTokenizerOpModel(const std::string& model_buffer,
                                const std::vector<std::string>& input_strings,
                                bool add_bos = false, bool add_eos = false,
                                bool reverse = false) {
    BuildModelAndInterpreter(model_buffer, input_strings, add_bos, add_eos,
                             reverse);
  }

  ~SentencepieceTokenizerOpModel() {
    if (interpreter_ != nullptr) TfLiteInterpreterDelete(interpreter_);
    if (options_ != nullptr) TfLiteInterpreterOptionsDelete(options_);
    if (model_ != nullptr) TfLiteModelDelete(model_);
  }

  TfLiteStatus Invoke() {
    if (interpreter_ == nullptr) return kTfLiteError;
    return TfLiteInterpreterInvoke(interpreter_);
  }

  // Returns the token IDs of all input strings, concatenated (output 0).
  std::vector<int32_t> GetOutputValues() { return GetOutputIntVector(0); }

  // Returns the row offsets into `GetOutputValues()` (output 1).
  std::vector<int32_t> GetOutputSplits() { return GetOutputIntVector(1); }

 private:
  // Copies the contents of the INT32 output tensor at `output_tensor_index`
  // into a vector. Returns an empty vector if the tensor is missing or empty.
  std::vector<int32_t> GetOutputIntVector(int output_tensor_index) {
    const TfLiteTensor* output_tensor =
        TfLiteInterpreterGetOutputTensor(interpreter_, output_tensor_index);
    if (output_tensor == nullptr) return {};
    const size_t num_elements =
        TfLiteTensorByteSize(output_tensor) / sizeof(int32_t);
    const int32_t* data =
        static_cast<const int32_t*>(TfLiteTensorData(output_tensor));
    if (data == nullptr || num_elements == 0) return {};
    return std::vector<int32_t>(data, data + num_elements);
  }

  // Builds an in-memory TFLite model with a single TFSentencepieceTokenizeOp,
  // where `model_buffer`, `input_strings` and the flags are all constant
  // tensors. Then creates the interpreter and allocates tensors so the model is
  // ready to `Invoke()`.
  void BuildModelAndInterpreter(const std::string& model_buffer,
                                const std::vector<std::string>& input_strings,
                                bool add_bos, bool add_eos, bool reverse) {
    flatbuffers::FlatBufferBuilder builder;

    // Buffer 0 is reserved for empty/non-constant tensors per TFLite schema.
    std::vector<flatbuffers::Offset<Buffer>> buffers;
    buffers.push_back(
        CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>())));

    // Buffer 1: The serialized SentencePiece model bytes stored as a buffer
    // tensor.
    buffers.push_back(CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(model_buffer.data()),
                     model_buffer.size())));

    // Buffer 2: nbest (int32_t = 0).
    const int32_t nbest_val = 0;
    buffers.push_back(CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(&nbest_val),
                             sizeof(int32_t))));

    // Buffer 3: alpha (float = 0.0f).
    const float alpha_val = 0.0f;
    buffers.push_back(CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(&alpha_val),
                             sizeof(float))));

    // Buffer 4: add_bos flag.
    const bool add_bos_val = add_bos;
    buffers.push_back(CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(&add_bos_val),
                             sizeof(bool))));

    // Buffer 5: add_eos flag.
    const bool add_eos_val = add_eos;
    buffers.push_back(CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(&add_eos_val),
                             sizeof(bool))));

    // Buffer 6: reverse flag.
    const bool reverse_val = reverse;
    buffers.push_back(CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(&reverse_val),
                             sizeof(bool))));

    // Buffer 7: the input strings, serialized in the TFLite STRING tensor
    // layout. The layout is read as int32_t values, so the vector is aligned
    // to keep those reads aligned.
    const std::vector<uint8_t> text_bytes =
        SerializeStringTensorBuffer(input_strings);
    builder.ForceVectorAlignment(text_bytes.size(), sizeof(uint8_t),
                                 /*alignment=*/16);
    buffers.push_back(CreateBuffer(builder, builder.CreateVector(text_bytes)));

    // Define tensors for the TFSentencepieceTokenizeOp operator.
    std::vector<flatbuffers::Offset<Tensor>> tensors;

    // Tensor 0: model buffer tensor (UINT8, constant buffer 1).
    const int32_t model_shape[1] = {static_cast<int32_t>(model_buffer.size())};
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(model_shape, 1),
        TensorType_UINT8, /*buffer=*/1, builder.CreateString("model")));

    // Tensor 1: text string tensor (STRING, constant buffer 7).
    const int32_t text_shape[1] = {static_cast<int32_t>(input_strings.size())};
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(text_shape, 1),
        TensorType_STRING, /*buffer=*/7, builder.CreateString("text")));

    // Tensor 2: nbest (INT32, constant buffer 2).
    const int32_t scalar_shape[1] = {1};
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(scalar_shape, 1),
        TensorType_INT32, /*buffer=*/2, builder.CreateString("nbest")));

    // Tensor 3: alpha (FLOAT32, constant buffer 3).
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(scalar_shape, 1),
        TensorType_FLOAT32, /*buffer=*/3, builder.CreateString("alpha")));

    // Tensor 4: add_bos (BOOL, constant buffer 4).
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(scalar_shape, 1),
        TensorType_BOOL, /*buffer=*/4, builder.CreateString("add_bos")));

    // Tensor 5: add_eos (BOOL, constant buffer 5).
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(scalar_shape, 1),
        TensorType_BOOL, /*buffer=*/5, builder.CreateString("add_eos")));

    // Tensor 6: reverse (BOOL, constant buffer 6).
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(scalar_shape, 1),
        TensorType_BOOL, /*buffer=*/6, builder.CreateString("reverse")));

    // Tensor 7: output_values (INT32, dynamically allocated during eval).
    const std::vector<int32_t> empty_shape;
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(empty_shape), TensorType_INT32,
        /*buffer=*/0, builder.CreateString("output_values")));

    // Tensor 8: output_splits (INT32, dynamically allocated during eval).
    tensors.push_back(CreateTensor(
        builder, builder.CreateVector<int32_t>(empty_shape), TensorType_INT32,
        /*buffer=*/0, builder.CreateString("output_splits")));

    // Custom operator registration and opcode definition.
    flatbuffers::Offset<OperatorCode> op_code = CreateOperatorCodeDirect(
        builder, BuiltinOperator_CUSTOM, "TFSentencepieceTokenizeOp",
        /*version=*/1, BuiltinOperator_CUSTOM);

    const int32_t op_inputs[7] = {0, 1, 2, 3, 4, 5, 6};
    const int32_t op_outputs[2] = {7, 8};
    flatbuffers::Offset<Operator> op = CreateOperator(
        builder, /*opcode_index=*/0,
        builder.CreateVector<int32_t>(op_inputs, 7),
        builder.CreateVector<int32_t>(op_outputs, 2), BuiltinOptions_NONE,
        /*builtin_options=*/0,
        /*custom_options=*/0, CustomOptionsFormat_FLEXBUFFERS);

    // SubGraph definition: every op input is a constant tensor, so the subgraph
    // has no runtime inputs. Outputs are tensors 7 & 8.
    const int32_t subgraph_outputs[2] = {7, 8};
    flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
        builder, builder.CreateVector(tensors.data(), tensors.size()),
        builder.CreateVector<int32_t>(std::vector<int32_t>()),
        builder.CreateVector<int32_t>(subgraph_outputs, 2),
        builder.CreateVector(&op, 1), /*name=*/0);

    flatbuffers::Offset<Model> model_offset = CreateModel(
        builder, TfLiteSchemaVersion(), builder.CreateVector(&op_code, 1),
        builder.CreateVector(&subgraph, 1),
        builder.CreateString("sentencepiece_tokenizer_model"),
        builder.CreateVector(buffers.data(), buffers.size()));
    FinishModelBuffer(builder, model_offset);

    // Copy FlatBuffer bytes to model_data_ to ensure backing memory survives.
    model_data_.assign(
        reinterpret_cast<const char*>(builder.GetBufferPointer()),
        builder.GetSize());
    model_ = TfLiteModelCreate(model_data_.data(), model_data_.size());
    ASSERT_NE(model_, nullptr);

    options_ = TfLiteInterpreterOptionsCreate();
    ASSERT_NE(options_, nullptr);
    TfLiteInterpreterOptionsAddOperator(
        options_, RegisterExternal_SENTENCEPIECE_TOKENIZER());

    interpreter_ = TfLiteInterpreterCreate(model_, options_);
    ASSERT_NE(interpreter_, nullptr);
    ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter_), kTfLiteOk);
  }

  std::string model_data_;
  TfLiteModel* model_ = nullptr;
  TfLiteInterpreterOptions* options_ = nullptr;
  TfLiteInterpreter* interpreter_ = nullptr;
};

// Tokenizing a single string must produce exactly the golden IDs, with splits
// spanning all of them.
TEST_F(SentencepieceTokenizerOp, TokenizeSingleString) {
  SentencepieceTokenizerOpModel m(model(), {"Hello world!"}, /*add_bos=*/false,
                                  /*add_eos=*/false, /*reverse=*/false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputValues(), ElementsAreArray(kHelloWorldIds));
  EXPECT_THAT(m.GetOutputSplits(),
              ElementsAre(0, static_cast<int32_t>(std::size(kHelloWorldIds))));
}

// Tokenizing several strings concatenates their IDs in input order, and the
// splits mark where each string's IDs begin and end.
TEST_F(SentencepieceTokenizerOp, TokenizeMultipleStrings) {
  SentencepieceTokenizerOpModel m(model(), {"Hello world!", "world"},
                                  /*add_bos=*/false, /*add_eos=*/false,
                                  /*reverse=*/false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const std::vector<int32_t> splits = m.GetOutputSplits();
  const std::vector<int32_t> values = m.GetOutputValues();

  // The splits have one leading zero plus one end offset per input string.
  // This is fatal so the indexing below never reads out of bounds.
  ASSERT_THAT(splits, SizeIs(3));
  EXPECT_EQ(splits[0], 0);
  EXPECT_THAT(splits[0], Lt(splits[1]));
  EXPECT_THAT(splits[1], Lt(splits[2]));
  EXPECT_EQ(splits[2], static_cast<int32_t>(values.size()));

  // Each string's slice of `values` must match its golden IDs.
  std::vector<int32_t> expected_values(std::begin(kHelloWorldIds),
                                       std::end(kHelloWorldIds));
  expected_values.insert(expected_values.end(), std::begin(kWorldIds),
                         std::end(kWorldIds));
  EXPECT_THAT(values, ElementsAreArray(expected_values));
  EXPECT_EQ(splits[1], static_cast<int32_t>(std::size(kHelloWorldIds)));
}

// The `add_bos` / `add_eos` flags wrap the golden IDs in BOS and EOS tokens.
TEST_F(SentencepieceTokenizerOp, AddBosAndEos) {
  SentencepieceTokenizerOpModel m(model(), {"Hello world!"}, /*add_bos=*/true,
                                  /*add_eos=*/true, /*reverse=*/false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::vector<int32_t> expected_values = {kBosId};
  expected_values.insert(expected_values.end(), std::begin(kHelloWorldIds),
                         std::end(kHelloWorldIds));
  expected_values.push_back(kEosId);
  EXPECT_THAT(m.GetOutputValues(), ElementsAreArray(expected_values));
}

// Only `add_bos` set: BOS is prepended and no EOS is appended. Guards against
// the kernel reading the wrong flag input for either flag.
TEST_F(SentencepieceTokenizerOp, AddBosOnly) {
  SentencepieceTokenizerOpModel m(model(), {"Hello world!"}, /*add_bos=*/true,
                                  /*add_eos=*/false, /*reverse=*/false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  std::vector<int32_t> expected_values = {kBosId};
  expected_values.insert(expected_values.end(), std::begin(kHelloWorldIds),
                         std::end(kHelloWorldIds));
  EXPECT_THAT(m.GetOutputValues(), ElementsAreArray(expected_values));
}

// The `reverse` flag emits the golden IDs in reverse order.
TEST_F(SentencepieceTokenizerOp, ReverseTokens) {
  SentencepieceTokenizerOpModel m(model(), {"Hello world!"}, /*add_bos=*/false,
                                  /*add_eos=*/false, /*reverse=*/true);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  const std::vector<int32_t> expected_values(std::rbegin(kHelloWorldIds),
                                             std::rend(kHelloWorldIds));
  EXPECT_THAT(m.GetOutputValues(), ElementsAreArray(expected_values));
}

// An empty string yields no tokens but still contributes one (empty) row.
TEST_F(SentencepieceTokenizerOp, EmptyInputString) {
  SentencepieceTokenizerOpModel m(model(), {""}, /*add_bos=*/false,
                                  /*add_eos=*/false, /*reverse=*/false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputSplits(), ElementsAre(0, 0));
  EXPECT_THAT(m.GetOutputValues(), IsEmpty());
}

// An input tensor with no strings (shape [0]) yields no rows: the splits hold
// only the leading zero and there are no values.
TEST_F(SentencepieceTokenizerOp, ZeroInputStrings) {
  SentencepieceTokenizerOpModel m(model(), /*input_strings=*/{},
                                  /*add_bos=*/false, /*add_eos=*/false,
                                  /*reverse=*/false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);

  EXPECT_THAT(m.GetOutputSplits(), ElementsAre(0));
  EXPECT_THAT(m.GetOutputValues(), IsEmpty());
}

// The outputs are dynamic tensors that the kernel resizes on every `Invoke()`.
// A second `Invoke()` must re-resize them and produce identical results.
TEST_F(SentencepieceTokenizerOp, RepeatedInvokeProducesSameOutput) {
  SentencepieceTokenizerOpModel m(model(), {"Hello world!", "world"},
                                  /*add_bos=*/false, /*add_eos=*/false,
                                  /*reverse=*/false);
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  const std::vector<int32_t> first_values = m.GetOutputValues();
  const std::vector<int32_t> first_splits = m.GetOutputSplits();

  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_EQ(m.GetOutputValues(), first_values);
  EXPECT_EQ(m.GetOutputSplits(), first_splits);
  EXPECT_THAT(first_splits, SizeIs(3));
}

// Both registration entry points return a usable registration, and the legacy
// one wraps the opaque-API registration.
TEST_F(SentencepieceTokenizerOp, ExternalRegistration) {
  EXPECT_NE(RegisterExternal_SENTENCEPIECE_TOKENIZER(), nullptr);
  EXPECT_NE(Register_SENTENCEPIECE_TOKENIZER(), nullptr);
  EXPECT_NE(Register_SENTENCEPIECE_TOKENIZER()->registration_external, nullptr);
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite
