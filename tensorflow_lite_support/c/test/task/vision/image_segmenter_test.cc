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

#include "tensorflow_lite_support/c/task/vision/image_segmenter.h"

#include <string.h>

#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow_lite_support/c/common.h"
#include "tensorflow_lite_support/c/task/processor/segmentation_result.h"
#include "tensorflow_lite_support/c/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/port/status_matchers.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

#define GTEST_COUT std::cerr << "[          ] [ INFO ]"


namespace tflite {
namespace task {
namespace vision {
namespace {

using ::testing::HasSubstr;
using ::tflite::support::StatusOr;
using ::tflite::task::JoinPath;

constexpr char kTestDataDirectory[] =
    "/tensorflow_lite_support/cc/test/testdata/task/"
    "vision/";
// Quantized model.
constexpr char kDeepLabV3[] = "deeplabv3.tflite";

StatusOr<ImageData> LoadImage(const char* image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

class ImageSegmenterFromOptionsTest : public tflite_shims::testing::Test {};

// TEST_F(ImageClassifierFromOptionsTest, FailsWithNullOptionsAndError) {
//   TfLiteSupportError* error = nullptr;
//   TfLiteImageClassifier* image_classifier =
//       TfLiteImageClassifierFromOptions(nullptr, &error);

//   EXPECT_EQ(image_classifier, nullptr);
//   if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

//   ASSERT_NE(error, nullptr);
//   EXPECT_EQ(error->code, kInvalidArgumentError);
//   EXPECT_NE(error->message, nullptr);
//   EXPECT_THAT(error->message, HasSubstr("Expected non null options"));

//   TfLiteSupportErrorDelete(error);
// }

// TEST_F(ImageClassifierFromOptionsTest, FailsWithMissingModelPath) {
//   TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
//   TfLiteImageClassifier* image_classifier =
//       TfLiteImageClassifierFromOptions(&options, nullptr);
//   EXPECT_EQ(image_classifier, nullptr);
//   if (image_classifier) TfLiteImageClassifierDelete(image_classifier);
// }

// TEST_F(ImageClassifierFromOptionsTest, FailsWithMissingModelPathAndError) {
//   TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();

//   TfLiteSupportError* error = nullptr;
//   TfLiteImageClassifier* image_classifier =
//       TfLiteImageClassifierFromOptions(&options, &error);

//   EXPECT_EQ(image_classifier, nullptr);
//   if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

//   ASSERT_NE(error, nullptr);
//   EXPECT_EQ(error->code, kInvalidArgumentError);
//   EXPECT_NE(error->message, nullptr);
//   EXPECT_THAT(error->message, HasSubstr("`base_options.model_file`"));

//   TfLiteSupportErrorDelete(error);
// }

TEST_F(ImageSegmenterFromOptionsTest, SucceedsWithModelPath) {
    GTEST_COUT << "In teest" << std::endl;
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kDeepLabV3);
  TfLiteImageSegmenterOptions options = TfLiteImageSegmenterOptionsCreate();
  GTEST_COUT << "Created Opt" << std::endl;

  options.base_options.model_file.file_path = model_path.data();
  GTEST_COUT << "ModelPath" << std::endl;
  TfLiteImageSegmenter* image_segmenter =
      TfLiteImageSegmenterFromOptions(&options, nullptr);
  
  GTEST_COUT << "Seggmeenteer Create" << std::endl;
  EXPECT_NE(image_segmenter, nullptr);
  TfLiteImageSegmenterDelete(image_segmenter);
}

TEST_F(ImageSegmenterFromOptionsTest, SucceedsWithNumberOfThreadsAndError) {
  std::string model_path =
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kDeepLabV3);
  TfLiteImageSegmenterOptions options = TfLiteImageSegmenterOptionsCreate();
  options.base_options.model_file.file_path = model_path.data();
  options.base_options.compute_settings.cpu_settings.num_threads = 3;

  TfLiteSupportError* error = nullptr;
  TfLiteImageSegmenter* image_segmenter =
      TfLiteImageSegmenterFromOptions(&options, &error);

  EXPECT_NE(image_segmenter, nullptr);
  EXPECT_EQ(error, nullptr);

  if (image_segmenter) TfLiteImageSegmenterDelete(image_segmenter);
  if (error) TfLiteSupportErrorDelete(error);
}

// TEST(ImageClassifierNullClassifierClassifyTest,
//      FailsWithNullImageClassifierAndError) {
//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

//   TfLiteSupportError* error = nullptr;
//   TfLiteClassificationResult* classification_result =
//       TfLiteImageClassifierClassify(nullptr, nullptr, &error);

//   ImageDataFree(&image_data);

//   EXPECT_EQ(classification_result, nullptr);
//   if (classification_result)
//     TfLiteClassificationResultDelete(classification_result);

//   ASSERT_NE(error, nullptr);
//   EXPECT_EQ(error->code, kInvalidArgumentError);
//   EXPECT_NE(error->message, nullptr);
//   EXPECT_THAT(error->message, HasSubstr("Expected non null image classifier"));

//   TfLiteSupportErrorDelete(error);
// }

class ImageSegmenterSegmentTest : public tflite_shims::testing::Test {
 protected:
  void SetUp() override {
    std::string model_path =
        JoinPath("./" /*test src dir*/, kTestDataDirectory,
                 kDeepLabV3);

    TfLiteImageSegmenterOptions options = TfLiteImageSegmenterOptionsCreate();
    options.base_options.model_file.file_path = model_path.data();
    image_classifier = TfLiteImageSegmenterFromOptions(&options, nullptr);
    ASSERT_NE(image_classifier, nullptr);
  }

  void TearDown() override { TfLiteImageSegmenterDelete(image_classifier); }
  TfLiteImageSegmenter* image_classifier;
};

TEST_F(ImageSegmenterSegmentTest, SucceedsWithImageData) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger_crop.jpg"));

  TfLiteFrameBuffer frame_buffer = {
      .format = kRGB,
      .orientation = kTopLeft,
      .dimension = {.width = image_data.width, .height = image_data.height},
      .buffer = image_data.pixel_data};

  TfLiteSegmentationResult* segmentation_result = TfLiteImageSegmenterSegment(image_classifier, &frame_buffer, nullptr);

  ImageDataFree(&image_data);
  GTEST_COUT << "Done" << std::endl;

  // ASSERT_NE(classification_result, nullptr);
  // EXPECT_GE(classification_result->size, 1);
  // EXPECT_NE(classification_result->classifications, nullptr);
  // EXPECT_GE(classification_result->classifications->size, 1);
  // EXPECT_NE(classification_result->classifications->categories, nullptr);
  // EXPECT_EQ(strcmp(classification_result->classifications->categories[0].label,
  //                  "cheeseburger"),
  //           0);
  // EXPECT_GE(classification_result->classifications->categories[0].score, 0.90);

  // TfLiteClassificationResultDelete(classification_result);
}

// TEST_F(ImageClassifierClassifyTest, FailsWithNullFrameBufferAndError) {
//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

//   TfLiteSupportError* error = nullptr;
//   TfLiteClassificationResult* classification_result =
//       TfLiteImageClassifierClassify(image_classifier, nullptr, &error);

//   ImageDataFree(&image_data);

//   EXPECT_EQ(classification_result, nullptr);
//   if (classification_result)
//     TfLiteClassificationResultDelete(classification_result);

//   ASSERT_NE(error, nullptr);
//   EXPECT_EQ(error->code, kInvalidArgumentError);
//   EXPECT_NE(error->message, nullptr);
//   EXPECT_THAT(error->message, HasSubstr("Expected non null frame buffer"));

//   TfLiteSupportErrorDelete(error);
// }

// TEST_F(ImageClassifierClassifyTest, FailsWithNullImageDataAndError) {
//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

//   TfLiteFrameBuffer frame_buffer = {.format = kRGB, .orientation = kTopLeft};

//   TfLiteSupportError* error = nullptr;
//   TfLiteClassificationResult* classification_result =
//       TfLiteImageClassifierClassify(image_classifier, &frame_buffer, &error);

//   ImageDataFree(&image_data);

//   EXPECT_EQ(classification_result, nullptr);
//   if (classification_result)
//     TfLiteClassificationResultDelete(classification_result);

//   ASSERT_NE(error, nullptr);
//   EXPECT_EQ(error->code, kInvalidArgumentError);
//   EXPECT_NE(error->message, nullptr);
//   EXPECT_THAT(error->message, HasSubstr("Invalid stride information"));

//   TfLiteSupportErrorDelete(error);
// }

// TEST_F(ImageClassifierClassifyTest, SucceedsWithRoiWithinImageBounds) {
//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

//   TfLiteFrameBuffer frame_buffer = {
//       .format = kRGB,
//       .orientation = kTopLeft,
//       .dimension = {.width = image_data.width, .height = image_data.height},
//       .buffer = image_data.pixel_data};

//   TfLiteBoundingBox bounding_box = {
//       .origin_x = 0, .origin_y = 0, .width = 100, .height = 100};
//   TfLiteSupportError* error = nullptr;
//   TfLiteClassificationResult* classification_result =
//       TfLiteImageClassifierClassifyWithRoi(image_classifier, &frame_buffer,
//                                            &bounding_box, &error);

//   ImageDataFree(&image_data);

//   ASSERT_NE(classification_result, nullptr);
//   EXPECT_GE(classification_result->size, 1);
//   EXPECT_NE(classification_result->classifications, nullptr);
//   EXPECT_GE(classification_result->classifications->size, 1);
//   EXPECT_NE(classification_result->classifications->categories, nullptr);
//   EXPECT_EQ(strcmp(classification_result->classifications->categories[0].label,
//                    "bagel"),
//             0);
//   EXPECT_GE(classification_result->classifications->categories[0].score, 0.30);

//   TfLiteClassificationResultDelete(classification_result);
// }

// TEST_F(ImageClassifierClassifyTest, FailsWithRoiOutsideImageBoundsAndError) {
//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

//   TfLiteFrameBuffer frame_buffer = {
//       .format = kRGB,
//       .orientation = kTopLeft,
//       .dimension = {.width = image_data.width, .height = image_data.height},
//       .buffer = image_data.pixel_data};

//   TfLiteBoundingBox bounding_box = {
//       .origin_x = 0, .origin_y = 0, .width = 250, .height = 250};
//   TfLiteSupportError* error = nullptr;
//   TfLiteClassificationResult* classification_result =
//       TfLiteImageClassifierClassifyWithRoi(image_classifier, &frame_buffer,
//                                            &bounding_box, &error);

//   ImageDataFree(&image_data);

//   EXPECT_EQ(classification_result, nullptr);
//   if (classification_result)
//     TfLiteClassificationResultDelete(classification_result);

//   ASSERT_NE(error, nullptr);
//   EXPECT_EQ(error->code, kInvalidArgumentError);
//   EXPECT_NE(error->message, nullptr);
//   EXPECT_THAT(error->message, HasSubstr("Invalid crop coordinates"));

//   TfLiteSupportErrorDelete(error);
// }

// TEST(ImageClassifierWithUserDefinedOptionsClassifyTest,
//      SucceedsWithClassNameDenyList) {
//   char* denylisted_label_name = (char*)"cheeseburger";
//   std::string model_path =
//       JoinPath("./" /*test src dir*/, kTestDataDirectory,
//                kMobileNetQuantizedWithMetadata);

//   TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
//   options.base_options.model_file.file_path = model_path.data();

//   char* label_denylist[12] = {denylisted_label_name};
//   options.classification_options.label_denylist.list = label_denylist;
//   options.classification_options.label_denylist.length = 1;

//   TfLiteImageClassifier* image_classifier =
//       TfLiteImageClassifierFromOptions(&options, nullptr);
//   ASSERT_NE(image_classifier, nullptr);

//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

//   TfLiteFrameBuffer frame_buffer = {
//       .format = kRGB,
//       .orientation = kTopLeft,
//       .dimension = {.width = image_data.width, .height = image_data.height},
//       .buffer = image_data.pixel_data};

//   TfLiteClassificationResult* classification_result =
//       TfLiteImageClassifierClassify(image_classifier, &frame_buffer, nullptr);

//   ImageDataFree(&image_data);
//   if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

//   ASSERT_NE(classification_result, nullptr);
//   EXPECT_GE(classification_result->size, 1);
//   EXPECT_NE(classification_result->classifications, nullptr);
//   EXPECT_GE(classification_result->classifications->size, 1);
//   EXPECT_NE(classification_result->classifications->categories, nullptr);
//   EXPECT_NE(strcmp(classification_result->classifications->categories[0].label,
//                    denylisted_label_name),
//             0);

//   TfLiteClassificationResultDelete(classification_result);
// }

// TEST(ImageClassifierWithUserDefinedOptionsClassifyTest,
//      SucceedsWithClassNameAllowList) {
//   char* allowlisted_label_name = (char*)"cheeseburger";
//   std::string model_path =
//       JoinPath("./" /*test src dir*/, kTestDataDirectory,
//                kMobileNetQuantizedWithMetadata)
//           .data();

//   TfLiteImageClassifierOptions options = TfLiteImageClassifierOptionsCreate();
//   options.base_options.model_file.file_path = model_path.data();

//   char* label_allowlist[12] = {allowlisted_label_name};
//   options.classification_options.label_allowlist.list = label_allowlist;
//   options.classification_options.label_allowlist.length = 1;

//   TfLiteImageClassifier* image_classifier =
//       TfLiteImageClassifierFromOptions(&options, nullptr);
//   ASSERT_NE(image_classifier, nullptr);

//   SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData image_data, LoadImage("burger-224.png"));

//   TfLiteFrameBuffer frame_buffer = {
//       .format = kRGB,
//       .orientation = kTopLeft,
//       .dimension = {.width = image_data.width, .height = image_data.height},
//       .buffer = image_data.pixel_data};

//   TfLiteClassificationResult* classification_result =
//       TfLiteImageClassifierClassify(image_classifier, &frame_buffer, nullptr);

//   ImageDataFree(&image_data);
//   if (image_classifier) TfLiteImageClassifierDelete(image_classifier);

//   ASSERT_NE(classification_result, nullptr);
//   EXPECT_GE(classification_result->size, 1);
//   EXPECT_NE(classification_result->classifications, nullptr);
//   EXPECT_GE(classification_result->classifications->size, 1);
//   EXPECT_NE(classification_result->classifications->categories, nullptr);
//   EXPECT_EQ(strcmp(classification_result->classifications->categories[0].label,
//                    allowlisted_label_name),
//             0);

//   TfLiteClassificationResultDelete(classification_result);
// }

}  // namespace
}  // namespace vision
}  // namespace task
}  // namespace tflite
