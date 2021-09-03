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

#include <fstream>
#include <memory>

#include "absl/status/status.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/port/status_matchers.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

namespace tflite {
namespace task {
namespace processor {
namespace {

using ::tflite::support::StatusOr;
using ::tflite::task::JoinPath;
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

    SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image, LoadImage("burger.jpg"));
    frame_buffer_ = CreateFromRgbRawBuffer(
        image.pixel_data, FrameBuffer::Dimension{image.width, image.height});

    preprocessor_->Preprocess(*frame_buffer_);
  }

 protected:
  std::unique_ptr<TfLiteEngine> engine_ = nullptr;
  std::unique_ptr<FrameBuffer> frame_buffer_ = nullptr;
  std::unique_ptr<ImagePreprocessor> preprocessor_ = nullptr;
};

// See if output tensor has been re-dimmed as per the input
// tensor. Expected shape: (1, input_height, input_width, 16).
TEST_F(DynamicInputTest, OutputDimensionCheck) {
  EXPECT_TRUE(engine_->interpreter_wrapper()->InvokeWithoutFallback().ok());
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
  // Get the processed input image.
  float *processed_input_data =
      tflite::task::core::AssertAndReturnTypedTensor<float>(
          engine_->GetInputs()[0]);

  bool is_equal = true;

  const uint8* image_data = frame_buffer_->plane(0).buffer;
  const size_t input_byte_size = frame_buffer_->plane(0).stride.row_stride_bytes *
                           frame_buffer_->dimension().height;

  for (size_t i = 0; i < input_byte_size / sizeof(uint8);
       ++i, ++image_data, ++processed_input_data)
    is_equal &=
        std::fabs(static_cast<float>(*image_data) - *processed_input_data) <=
        std::numeric_limits<float>::epsilon();

  EXPECT_TRUE(is_equal);
}
}  // namespace
}  // namespace processor
}  // namespace task
}  // namespace tflite
