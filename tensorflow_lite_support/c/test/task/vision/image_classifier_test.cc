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

#include "tensorflow_lite_support/c/task/vision/image_classifier.h"

#include <string.h>

#include "tensorflow_lite_support/c/task/processor/classification_result.h"
#include "tensorflow_lite_support/c/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/c/test/task/vision/utils/image_utils.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"

namespace tflite {
namespace task {
namespace vision {
namespace {

using ::tflite::task::JoinPath;

constexpr char kTestDataDirectory[] =
    "tensorflow_lite_support/cc/test/testdata/task/vision/";
// Quantized model.
constexpr char kMobileNetQuantizedWithMetadata[] =
    "mobilenet_v1_0.25_224_quant.tflite";

CImageData LoadImage(const char* image_name) {
  return CDecodeImageFromFile(
      JoinPath("./" /*test src dir*/, kTestDataDirectory, image_name).data());
}

TEST(CImageClassifierFromOptionsTest, FailsWithMissingModelPath) {
  TfLiteImageClassifierOptions options = {0};
  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options);
  EXPECT_EQ(image_classifier, nullptr);
}

TEST(CImageClassifierFromOptionsTest, SucceedsWithModelPath) {
  //   TfLiteImageClassifierOptions* options =
  //   TfLiteImageClassifierOptionsCreate();
  const char* model_path = JoinPath("./" /*test src dir*/, kTestDataDirectory,
                                    kMobileNetQuantizedWithMetadata)
                               .data();
  TfLiteImageClassifierOptions options = {0};
  options.base_options.model_file.file_path = model_path;
  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options);
  EXPECT_NE(image_classifier, nullptr);
  TfLiteImageClassifierDelete(image_classifier);
}

TEST(CImageClassifierFromOptionsTest, SucceedsWithNumberOfThreads) {
  const char* model_path = JoinPath("./" /*test src dir*/, kTestDataDirectory,
                                    kMobileNetQuantizedWithMetadata)
                               .data();

  TfLiteImageClassifierOptions options = {0};
  options.base_options.model_file.file_path = model_path;
  options.base_options.compute_settings.cpu_settings.num_threads = 3;
  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options);
  EXPECT_NE(image_classifier, nullptr);
  TfLiteImageClassifierDelete(image_classifier);
}

TEST(CImageClassifierFromOptionsTest,
     FailsWithClassNameBlackListAndClassNameWhiteList) {
  const char* model_path = JoinPath("./" /*test src dir*/, kTestDataDirectory,
                                    kMobileNetQuantizedWithMetadata)
                               .data();

  TfLiteImageClassifierOptions options = {0};
  options.base_options.model_file.file_path = model_path;

  char* class_name_blacklist[] = {"brambling"};
  options.classification_options.class_name_blacklist.list =
      class_name_blacklist;
  options.classification_options.class_name_blacklist.length = 1;

  char* class_name_whitelist[] = {"cheeseburger"};
  options.classification_options.class_name_whitelist.list =
      class_name_whitelist;
  options.classification_options.class_name_whitelist.length = 1;

  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options);
  EXPECT_EQ(image_classifier, nullptr);
  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);
}

class CImageClassifierClassifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TfLiteImageClassifierOptions options = {0};
    options.base_options.model_file.file_path =
        JoinPath("./" /*test src dir*/, kTestDataDirectory,
                 kMobileNetQuantizedWithMetadata)
            .data();
    image_classifier = TfLiteImageClassifierFromOptions(&options);

    ASSERT_NE(image_classifier, nullptr);
  }

  void TearDown() override { TfLiteImageClassifierDelete(image_classifier); }
  TfLiteImageClassifier* image_classifier;
};

TEST_F(CImageClassifierClassifyTest, SucceedsWithImageData) {
  CImageData image_data = LoadImage("burger-224.png");

  TfLiteFrameBuffer frame_buffer = {.dimension.width = image_data.width,
                                    .dimension.height = image_data.height,
                                    .buffer = image_data.pixel_data,
                                    .format = kRGB};

  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(image_classifier, &frame_buffer);

  CImageDataFree(&image_data);

  ASSERT_NE(classification_result, nullptr);
  EXPECT_TRUE(classification_result->size >= 1);
  EXPECT_NE(classification_result->classifications, nullptr);
  EXPECT_TRUE(classification_result->classifications->size >= 1);
  EXPECT_NE(classification_result->classifications->categories, nullptr);
  // TODO(prianka): check score and labels`

  TfLiteClassificationResultDelete(classification_result);
}

TEST_F(CImageClassifierClassifyTest, SucceedsWithRoiWithinImageBounds) {
  CImageData image_data = LoadImage("burger-224.png");

  TfLiteFrameBuffer frame_buffer = {.dimension.width = image_data.width,
                                    .dimension.height = image_data.height,
                                    .buffer = image_data.pixel_data,
                                    .format = kRGB};

  TfLiteBoundingBox bounding_box = {
      .origin_x = 0, .origin_y = 0, .width = 100, .height = 100};
  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassifyWithRoi(image_classifier, &frame_buffer,
                                           &bounding_box);

  CImageDataFree(&image_data);

  ASSERT_NE(classification_result, nullptr);
  EXPECT_TRUE(classification_result->size >= 1);
  EXPECT_NE(classification_result->classifications, nullptr);
  EXPECT_TRUE(classification_result->classifications->size >= 1);
  EXPECT_NE(classification_result->classifications->categories, nullptr);
  // TODO(prianka): check score and labels`

  TfLiteClassificationResultDelete(classification_result);
}

TEST_F(CImageClassifierClassifyTest, FailsWithRoiOutsideImageBounds) {
  CImageData image_data = LoadImage("burger-224.png");

  TfLiteFrameBuffer frame_buffer = {.dimension.width = image_data.width,
                                    .dimension.height = image_data.height,
                                    .buffer = image_data.pixel_data,
                                    .format = kRGB};

  TfLiteBoundingBox bounding_box = {
      .origin_x = 0, .origin_y = 0, .width = 250, .height = 250};
  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassifyWithRoi(image_classifier, &frame_buffer,
                                           &bounding_box);

  CImageDataFree(&image_data);

  EXPECT_EQ(classification_result, nullptr);

  if (classification_result != nullptr)
    TfLiteClassificationResultDelete(classification_result);
}

TEST(CImageClassifierWithUserDefinedOptionsClassifyTest,
     SucceedsWithClassNameBlackList) {
  char* blacklisted_label_name = "cheeseburger";
  const char* model_path = JoinPath("./" /*test src dir*/, kTestDataDirectory,
                                    kMobileNetQuantizedWithMetadata)
                               .data();

  TfLiteImageClassifierOptions options = {0};
  options.base_options.model_file.file_path = model_path;

  char* class_name_blacklist[] = {blacklisted_label_name};
  options.classification_options.class_name_blacklist.list =
      class_name_blacklist;
  options.classification_options.class_name_blacklist.length = 1;

  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options);
  ASSERT_NE(image_classifier, nullptr);

  CImageData image_data = LoadImage("burger-224.png");

  TfLiteFrameBuffer frame_buffer = {.dimension.width = image_data.width,
                                    .dimension.height = image_data.height,
                                    .buffer = image_data.pixel_data,
                                    .format = kRGB};

  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(image_classifier, &frame_buffer);

  CImageDataFree(&image_data);

  ASSERT_NE(classification_result, nullptr);
  EXPECT_TRUE(classification_result->size >= 1);
  EXPECT_NE(classification_result->classifications, nullptr);
  EXPECT_TRUE(classification_result->classifications->size >= 1);
  EXPECT_NE(classification_result->classifications->categories, nullptr);
  EXPECT_TRUE(
      strcmp(classification_result->classifications->categories[0].label,
             blacklisted_label_name) != 0);

  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

  TfLiteClassificationResultDelete(classification_result);
}

TEST(CImageClassifierWithUserDefinedOptionsClassifyTest,
     SucceedsWithClassNameWhiteList) {
  char* whitelisted_label_name = "cheeseburger";
  const char* model_path = JoinPath("./" /*test src dir*/, kTestDataDirectory,
                                    kMobileNetQuantizedWithMetadata)
                               .data();

  TfLiteImageClassifierOptions options = {0};
  options.base_options.model_file.file_path = model_path;

  char* class_name_whitelist[] = {"cheeseburger"};
  options.classification_options.class_name_whitelist.list =
      class_name_whitelist;
  options.classification_options.class_name_whitelist.length = 1;

  TfLiteImageClassifier* image_classifier =
      TfLiteImageClassifierFromOptions(&options);
  ASSERT_NE(image_classifier, nullptr);

  CImageData image_data = LoadImage("burger-224.png");

  TfLiteFrameBuffer frame_buffer = {.dimension.width = image_data.width,
                                    .dimension.height = image_data.height,
                                    .buffer = image_data.pixel_data,
                                    .format = kRGB};

  TfLiteClassificationResult* classification_result =
      TfLiteImageClassifierClassify(image_classifier, &frame_buffer);

  CImageDataFree(&image_data);

  ASSERT_NE(classification_result, nullptr);
  EXPECT_TRUE(classification_result->size >= 1);
  EXPECT_NE(classification_result->classifications, nullptr);
  EXPECT_TRUE(classification_result->classifications->size == 1);
  EXPECT_NE(classification_result->classifications->categories, nullptr);
  EXPECT_TRUE(
      strcmp(classification_result->classifications->categories[0].label,
             whitelisted_label_name) == 0);

  if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

  TfLiteClassificationResultDelete(classification_result);
}

}  // namespace
}  // namespace vision
}  // namespace task
}  // namespace tflite
