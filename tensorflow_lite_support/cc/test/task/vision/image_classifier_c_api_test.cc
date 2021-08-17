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

#include "tensorflow_lite_support/cc/task/vision/image_classifier_c_api.h"


#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils_c.h"
#include "tensorflow_lite_support/cc/task/vision/classification_result_c_api.h"
#include "tensorflow_lite_support/cc/task/vision/image_classifier_c_api.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer_c_api.h"


namespace tflite {
namespace task {
namespace vision {
namespace {

using ::tflite::task::JoinPath;


constexpr char kTestDataDirectory[] =
    "tensorflow_lite_support/cc/test/testdata/task/vision/";
// Float model.
constexpr char kMobileNetFloatWithMetadata[] = "mobilenet_v2_1.0_224.tflite";
// Quantized model.
constexpr char kMobileNetQuantizedWithMetadata[] =
    "mobilenet_v1_0.25_224_quant.tflite";
// Hello world flowers classifier supporting 5 classes (quantized model).
constexpr char kAutoMLModelWithMetadata[] = "automl_labeler_model.tflite";

ImageData LoadImage(const char* image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name).data());
}

TEST(ImageClassifierFromFileTest, FailsWithMissingModelPath) {
//   ImageClassifierOptions *options = ImageClassifierOptionsCreate();

  ImageClassifier *image_classifier = ImageClassifierFromFile("");
  ASSERT_EQ(image_classifier, nullptr);
}

TEST(ImageClassifierFromFileTest, SucceedsWithModelPath) {  
  ImageClassifier *image_classifier = ImageClassifierFromFile(JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata).data());
  EXPECT_NE(image_classifier, nullptr);
  ImageClassifierDelete(image_classifier);
}

TEST(ImageClassifierFromOptionsTest, FailsWithMissingModelPath) {
  ImageClassifierOptions *options = ImageClassifierOptionsCreate();
  ImageClassifier *image_classifier = ImageClassifierFromOptions(options);
  ASSERT_EQ(image_classifier, nullptr);
}

TEST(ImageClassifierFromOptionsTest, SucceedsWithModelPath) {
  ImageClassifierOptions *options = ImageClassifierOptionsCreate();
  const char *model_path = JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata).data();
  
  ImageClassifierOptionsSetModelFilePath(options, model_path);
  ImageClassifier *image_classifier = ImageClassifierFromOptions(options);
  EXPECT_NE(image_classifier, nullptr);
  ImageClassifierDelete(image_classifier);
}

class ImageClassifierClassifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
     image_classifier = ImageClassifierFromFile(JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata).data());
     ASSERT_NE(image_classifier, nullptr);
  }

  void TearDown() override {
     ImageClassifierDelete(image_classifier);
  }

  ImageClassifier *image_classifier;
};
  
TEST_F(ImageClassifierClassifyTest, SucceedsWithModelPath) {  
  struct ImageData image_data = LoadImage("burger-224.png");

  struct FrameBuffer frame_buffer = {.dimension.width = image_data.width, 
                                     .dimension.height = image_data.width, 
                                     .plane.buffer = image_data.pixel_data, 
                                     .plane.stride.row_stride_bytes = image_data.width  * image_data.channels, 
                                     .plane.stride.pixel_stride_bytes = image_data.channels, 
                                     .format = kRGB};
  
  struct ClassificationResult *classification_result = ImageClassifierClassify(image_classifier, &frame_buffer);
  
  ImageDataFree(&image_data);
  
  ASSERT_NE(classification_result, nullptr) << "Classification Result is NULL";
  EXPECT_TRUE(classification_result->size >= 1) << "Classification Result size is 0";
  EXPECT_NE(classification_result->classifications, nullptr) << "Classification Result Classifications is NULL";
  EXPECT_TRUE(classification_result->classifications->size >= 1) << "Classification Result Classifications Size is NULL";
  EXPECT_NE(classification_result->classifications->classes, nullptr) << "Classification Result Classifications Classes is NULL";

  ImageClassifierClassificationResultDelete(classification_result);
}


}  // namespace
}  // namespace vision
}  // namespace task
}  // namespace tflite
