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

constexpr char kModelPath[] =
    "tensorflow_lite_support/cc/test/testdata/task/vision/dilated_conv.tflite";

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(
      JoinPath("./" /*test src dir*/, kTestDataDirectory, image_name));
}

class DynamicInputTest : public tflite_shims::testing::Test {
 public:
  void SetUp() {
    engine = absl::make_unique<TfLiteEngine>();
    engine->BuildModelFromFile(kModelPath);
    engine->InitInterpreter();

    SUPPORT_ASSERT_OK_AND_ASSIGN(preprocessor,
                                 ImagePreprocessor::Create(engine.get(), {0}));
  }

 protected:
  std::unique_ptr<ImagePreprocessor> preprocessor;
  std::unique_ptr<TfLiteEngine> engine;
};

// See if input tensor dims signature for height and width is -1
// because it is so in the model.
TEST_F(DynamicInputTest, InputHeightAndWidthMutable) {
  const TfLiteIntArray* input_dims_signature =
      engine->GetInputs()[0]->dims_signature;
  EXPECT_EQ(input_dims_signature->data[1], -1);
  EXPECT_EQ(input_dims_signature->data[2], -1);
}

// See if output tensor has been re-dimmed as per the input
// tensor.
TEST_F(DynamicInputTest, OutputHeightAndWidthMutable) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image, LoadImage("burger.jpg"));
  std::unique_ptr<FrameBuffer> image_frame_buffer = CreateFromRgbRawBuffer(
      image.pixel_data, FrameBuffer::Dimension{image.width, image.height});

  preprocessor->Preprocess(*image_frame_buffer);

  EXPECT_EQ(engine->GetInputs()[0]->dims->data[1],
            engine->GetOutputs()[0]->dims->data[1]);
  EXPECT_EQ(engine->GetInputs()[0]->dims->data[2],
            engine->GetOutputs()[0]->dims->data[2]);
}
}  // namespace
}  // namespace processor
}  // namespace task
}  // namespace tflite
