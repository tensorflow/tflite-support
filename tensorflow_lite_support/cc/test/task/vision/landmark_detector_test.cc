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

#include "tensorflow_lite_support/cc/task/vision/landmark_detector.h"

#include <memory>

#include "external/com_google_absl/absl/flags/flag.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/port/status_matchers.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/proto/landmark_detector_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/landmarks_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

namespace tflite {
namespace task {
namespace vision {
namespace {

using ::testing::HasSubstr;
using ::testing::Optional;
using ::tflite::support::kTfLiteSupportPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::JoinPath;

// Number of keypoints
const int NUM_KEYPOINTS = 17;

constexpr char kTestDataDirectory[] =
    "tensorflow_lite_support/cc/test/testdata/task/vision/";

// Float model.
constexpr char kMobileNetFloatModel[] =
    "lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite";

// List of expected y coordinates of each keypoint
constexpr float GOLDEN_KEY_Y[] = {
    0.319555, 0.303167, 0.307264, 0.307264, 0.307264, 0.348233,
    0.409685, 0.303167, 0.421976, 0.266296, 0.421976, 0.495719,
    0.548978, 0.340039, 0.749724, 0.270392, 0.921792};

// List of expected x coordinates of each keypoint
constexpr float GOLDEN_KEY_X[] = {
    0.430170, 0.446557, 0.442460, 0.475235, 0.483429, 0.589947,
    0.479332, 0.725143, 0.348233, 0.856242, 0.229424, 0.733337,
    0.684175, 0.884920, 0.721046, 0.872630, 0.708756};

// List of expected scores of each keypoint
constexpr float GOLDEN_SCORE[] = {
    0.753821, 0.802983, 0.499816, 0.569463, 0.499816, 0.843952,
    0.880824, 0.925889, 0.905405, 0.700562, 0.753821, 0.802983,
    0.880824, 0.635012, 0.843952, 0.843952, 0.958664};

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

class CreateFromOptionsTest : public tflite_shims::testing::Test {};

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  LandmarkDetectorOptions options;

  StatusOr<std::unique_ptr<LandmarkDetector>> landmark_detector_or =
      LandmarkDetector::CreateFromOptions(options);

  EXPECT_EQ(landmark_detector_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(
      landmark_detector_or.status().message(),
      HasSubstr("Missing mandatory `model_file` field in `base_options`"));
  EXPECT_THAT(landmark_detector_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

class DetectTest : public tflite_shims::testing::Test {};

TEST_F(DetectTest, SucceedsWithFloatModel) {
  SUPPORT_ASSERT_OK_AND_ASSIGN(ImageData rgb_image, LoadImage("girl.jpg"));
  std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      rgb_image.pixel_data,
      FrameBuffer::Dimension{rgb_image.width, rgb_image.height});

  LandmarkDetectorOptions options;
  options.mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetFloatModel));
  SUPPORT_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LandmarkDetector> landmark_detector,
                       LandmarkDetector::CreateFromOptions(options));

  SUPPORT_ASSERT_OK_AND_ASSIGN(LandmarkResult result,
                       landmark_detector->Detect(*frame_buffer));
  ImageDataFree(&rgb_image);

  for (int i = 0; i < NUM_KEYPOINTS; ++i) {
    EXPECT_NEAR(result.landmarks(i).position(0), GOLDEN_KEY_Y[i], 0.01);
    EXPECT_NEAR(result.landmarks(i).position(1), GOLDEN_KEY_X[i], 0.01);
    EXPECT_NEAR(result.landmarks(i).score(), GOLDEN_SCORE[i], 0.01);
  }
}

}  // namespace
}  // namespace vision
}  // namespace task
}  // namespace tflite
