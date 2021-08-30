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

#include "tensorflow_lite_support/cc/task/processor/image_preprocessor.h"

#include <memory>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/port/status_matchers.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"
#include <fstream>

namespace tflite {
namespace task {
namespace processor {
namespace {

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::tflite::support::kTfLiteSupportPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::JoinPath;
using ::tflite::task::core::PopulateTensor;
using ::tflite::task::core::TaskAPIFactory;
using ::tflite::task::core::TfLiteEngine;
using ::tflite::task::vision::DecodeImageFromFile;
using ::tflite::task::vision::FrameBuffer;
using ::tflite::task::vision::ImageData;

constexpr char kTestDataDirectory[] =
    "tensorflow_lite_support/cc/test/testdata/task/vision/";

constexpr char kDilatedConvolutionModelWithMetaData[] = "dilated_conv.tflite";

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(
      JoinPath("./" /*test src dir*/, kTestDataDirectory, image_name));
}

class DynamicInputTest : public tflite_shims::testing::Test {
public:
  void SetUp() {
    engine_ = absl::make_unique<TfLiteEngine>();
    engine_->BuildModelFromFile(JoinPath("./", kTestDataDirectory,
                                         kDilatedConvolutionModelWithMetaData));
    engine_->InitInterpreter();

    SUPPORT_ASSERT_OK_AND_ASSIGN(preprocessor_,
                                 ImagePreprocessor::Create(engine_.get(), {0}));
  }

protected:
  std::unique_ptr<ImagePreprocessor> preprocessor_ = nullptr;
  std::unique_ptr<TfLiteEngine> engine_ = nullptr;
};

// See if input tensor dims signature for height and width is -1
// because it is so in the model.
TEST_F(DynamicInputTest, InputHeightAndWidthMutable) {
  const TfLiteIntArray *input_dims_signature =
      engine_->GetInputs()[0]->dims_signature;
  EXPECT_EQ(input_dims_signature->data[1], -1);
  EXPECT_EQ(input_dims_signature->data[2], -1);
}

// See if output tensor has been re-dimmed as per the input
// tensor. Expected shape: (1, input_height, input_width, 16).
TEST_F(DynamicInputTest, OutputDimensionCheck) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image, LoadImage("burger.jpg"));
  std::unique_ptr<FrameBuffer> image_frame_buffer = CreateFromRgbRawBuffer(
      image.pixel_data, FrameBuffer::Dimension{image.width, image.height});

  preprocessor_->Preprocess(*image_frame_buffer);
  absl::Status status = engine_->interpreter_wrapper()->InvokeWithoutFallback();
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(engine_->GetOutputs()[0]->dims->data[0], 1);
  EXPECT_EQ(engine_->GetOutputs()[0]->dims->data[1],
            engine_->GetInputs()[0]->dims->data[1]);
  EXPECT_EQ(engine_->GetOutputs()[0]->dims->data[2],
            engine_->GetInputs()[0]->dims->data[2]);
  EXPECT_EQ(engine_->GetOutputs()[0]->dims->data[3], 16);
}

// Compare pre-processed input with an already pre-processed
// golden image.
TEST_F(DynamicInputTest, GoldenImageComparison) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image, LoadImage("burger.jpg"));
  std::unique_ptr<FrameBuffer> image_frame_buffer = CreateFromRgbRawBuffer(
      image.pixel_data, FrameBuffer::Dimension{image.width, image.height});

  preprocessor_->Preprocess(*image_frame_buffer);

  // Check the processed input image.
  float *processed_input_data =
      tflite::task::core::AssertAndReturnTypedTensor<float>(
          engine_->GetInputs()[0]);

  std::string file_path = JoinPath("./" /*test src dir*/, kTestDataDirectory,
                                   "burger_normalized.txt");

  std::ifstream golden_image(file_path);
  std::string curr_line;
  bool is_equal = true;
  float epsilon = 0.1f;

  while (std::getline(golden_image, curr_line)) {
    float val = std::stof(curr_line);
    is_equal &= std::fabs(val - *processed_input_data) <= epsilon;
    ++processed_input_data;
  }

  EXPECT_TRUE(is_equal);
}
} // namespace
} // namespace processor
} // namespace task
} // namespace tflite
