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

#include "tensorflow_lite_support/cc/task/vision/image_transformer.h"

#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
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
namespace vision {
namespace {

using ::tflite::support::StatusOr;
using ::tflite::task::JoinPath;
using ::tflite::task::core::TfLiteEngine;


constexpr char kTestDataDirectory[] =
    "/tensorflow_lite_support/cc/test/testdata/task/"
    "vision/";

constexpr char kESRGANModelWithInputAndOutputMetaData[] = "esrgan_with_input_and_output_metadata.tflite";
constexpr char kESRGANModelWithInputMetaData[] = "esrgan_with_input_metadata.tflite";

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

class PostprocessorTest : public tflite_shims::testing::Test {};

TEST_F(PostprocessorTest, FloatSucceedsWithFullMetadata) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData rgb_image, LoadImage("fox_downsampled.jpg"));

  std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      rgb_image.pixel_data,
      FrameBuffer::Dimension{rgb_image.width, rgb_image.height});
  ImageTransformerOptions options;
  options.mutable_base_options()->mutable_model_file()->set_file_name(
    JoinPath("./" /*test src dir*/, kTestDataDirectory,
              kESRGANModelWithInputAndOutputMetaData));
  SUPPORT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageTransformer> image_transformer,
                      ImageTransformer::CreateFromOptions(options));

  StatusOr<FrameBuffer> result_or =
      image_transformer->Transform(*frame_buffer);
  ImageDataFree(&rgb_image);
  SUPPORT_ASSERT_OK(result_or);
}

TEST_F(PostprocessorTest, FloatSucceedsWithPartialMetadata) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData rgb_image, LoadImage("fox_downsampled.jpg"));

  std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      rgb_image.pixel_data,
      FrameBuffer::Dimension{rgb_image.width, rgb_image.height});
  ImageTransformerOptions options;
  options.mutable_base_options()->mutable_model_file()->set_file_name(
    JoinPath("./" /*test src dir*/, kTestDataDirectory,
              kESRGANModelWithInputMetaData));
  SUPPORT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageTransformer> image_transformer,
                      ImageTransformer::CreateFromOptions(options));

  StatusOr<FrameBuffer> result_or =
      image_transformer->Transform(*frame_buffer);
  ImageDataFree(&rgb_image);
  SUPPORT_ASSERT_OK(result_or);
}

class SuperResolutionTest : public tflite_shims::testing::Test {};

// Calculate the peak signal-to-noise ratio.
// Original code: https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/.
double PSNR(const FrameBuffer& enhancedImage, const FrameBuffer& testImage) {
  int imageSize = testImage.dimension().width * testImage.dimension().height;
  const uint8* enhancedImagePtr = enhancedImage.plane(0).buffer;
  const uint8* testImagePtr = testImage.plane(0).buffer;
  double mse = 0.0;
  for (int i = 0; i < imageSize; ++i, ++enhancedImagePtr, ++testImagePtr) {
      mse += std::pow(static_cast<double>(*enhancedImagePtr) - static_cast<double>(*testImagePtr), 2);
  }
  mse /= imageSize;

  double maxPixel = 255.0;
  // Zero MSE means no noise is present in the signal.
  double psnr = mse == 0 ? 100.0 : 20 * std::log10(maxPixel / std::sqrt(mse));

  return psnr;
}

// Use a bi-cubically downsampled image as input to the model and compare
// the model output with the original image. Expected PSNR taken from:
// https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/image_enhancing.ipynb.
TEST_F(SuperResolutionTest, PSNRTest) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData fox_downsampled, LoadImage("fox_downsampled.jpg"));
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData fox_original, LoadImage("fox_original.jpg"));

  std::unique_ptr<FrameBuffer> fox_downsampled_buffer = CreateFromRgbRawBuffer(
      fox_downsampled.pixel_data,
      FrameBuffer::Dimension{fox_downsampled.width, fox_downsampled.height});

  std::unique_ptr<FrameBuffer> fox_original_buffer = CreateFromRgbRawBuffer(
      fox_original.pixel_data,
      FrameBuffer::Dimension{fox_original.width, fox_original.height});

  ImageTransformerOptions options;
  options.mutable_base_options()->mutable_model_file()->set_file_name(
    JoinPath("./" /*test src dir*/, kTestDataDirectory,
              kESRGANModelWithInputAndOutputMetaData));
  SUPPORT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ImageTransformer> image_transformer,
                      ImageTransformer::CreateFromOptions(options));

  StatusOr<FrameBuffer> result_or =
      image_transformer->Transform(*fox_downsampled_buffer);
  ImageDataFree(&fox_downsampled);
  ImageDataFree(&fox_original);
  SUPPORT_ASSERT_OK(result_or);
  EXPECT_DOUBLE_EQ(PSNR(*fox_original_buffer, result_or.value()), 28.029171);
}

}  // namespace
}  // namespace processor
}  // namespace task
}  // namespace tflite
