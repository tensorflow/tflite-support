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

#include "tensorflow_lite_support/cc/task/vision/object_detector.h"

#include <memory>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/shims/cc/shims_test_util.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/gmock.h"
#include "tensorflow_lite_support/cc/port/gtest.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/status_matchers.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/proto/bounding_box_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/detections_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/object_detector_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_utils.h"
#include "tensorflow_lite_support/cc/test/test_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

namespace tflite {

namespace ops {
namespace custom {

// Forward declaration for the custom Detection_PostProcess op.
//
// See:
// https://medium.com/@bsramasubramanian/running-a-tensorflow-lite-model-in-python-with-custom-ops-9b2b46efd355
TfLiteRegistration* Register_DETECTION_POSTPROCESS();

}  // namespace custom
}  // namespace ops

namespace task {
namespace vision {
namespace {

using ::testing::HasSubstr;
using ::testing::Optional;
using ::tflite::support::kTfLiteSupportPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::JoinPath;
using ::tflite::task::core::PopulateTensor;
using ::tflite::task::core::TaskAPIFactory;
using ::tflite::task::core::TfLiteEngine;

constexpr char kTestDataDirectory[] =
    "tensorflow_lite_support/cc/test/testdata/task/vision/";
constexpr char kMobileSsdWithMetadata[] =
    "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite";
constexpr char kExpectResults[] =
    R"pb(detections {
           bounding_box { origin_x: 54 origin_y: 396 width: 393 height: 196 }
           classes { index: 16 score: 0.64453125 class_name: "cat" }
         }
         detections {
           bounding_box { origin_x: 602 origin_y: 157 width: 394 height: 447 }
           classes { index: 16 score: 0.59765625 class_name: "cat" }
         }
         detections {
           bounding_box { origin_x: 261 origin_y: 394 width: 179 height: 209 }
           # Actually a dog, but the model gets confused.
           classes { index: 16 score: 0.5625 class_name: "cat" }
         }
         detections {
           bounding_box { origin_x: 389 origin_y: 197 width: 276 height: 409 }
           classes { index: 17 score: 0.51171875 class_name: "dog" }
         }
    )pb";

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

// OpResolver including the custom Detection_PostProcess op.
class MobileSsdQuantizedOpResolver : public ::tflite::MutableOpResolver {
 public:
  MobileSsdQuantizedOpResolver() {
    AddBuiltin(::tflite::BuiltinOperator_CONCATENATION,
               ::tflite::ops::builtin::Register_CONCATENATION());
    AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
               ::tflite::ops::builtin::Register_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_RESHAPE,
               ::tflite::ops::builtin::Register_RESHAPE());
    AddBuiltin(::tflite::BuiltinOperator_LOGISTIC,
               ::tflite::ops::builtin::Register_LOGISTIC());
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
    AddCustom("TFLite_Detection_PostProcess",
              tflite::ops::custom::Register_DETECTION_POSTPROCESS());
  }

  MobileSsdQuantizedOpResolver(const MobileSsdQuantizedOpResolver& r) = delete;
};

class CreateFromOptionsTest : public tflite_shims::testing::Test {};

TEST_F(CreateFromOptionsTest, SucceedsWithSelectiveOpResolver) {
  ObjectDetectorOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileSsdWithMetadata));

  SUPPORT_ASSERT_OK(ObjectDetector::CreateFromOptions(
      options, absl::make_unique<MobileSsdQuantizedOpResolver>()));
}

// OpResolver missing the Detection_PostProcess op.
class MobileSsdQuantizedOpResolverMissingOps
    : public ::tflite::MutableOpResolver {
 public:
  MobileSsdQuantizedOpResolverMissingOps() {
    AddBuiltin(::tflite::BuiltinOperator_CONCATENATION,
               ::tflite::ops::builtin::Register_CONCATENATION());
    AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
               ::tflite::ops::builtin::Register_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_RESHAPE,
               ::tflite::ops::builtin::Register_RESHAPE());
    AddBuiltin(::tflite::BuiltinOperator_LOGISTIC,
               ::tflite::ops::builtin::Register_LOGISTIC());
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
  }

  MobileSsdQuantizedOpResolverMissingOps(
      const MobileSsdQuantizedOpResolverMissingOps& r) = delete;
};

TEST_F(CreateFromOptionsTest, FailsWithSelectiveOpResolverMissingOps) {
  ObjectDetectorOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileSsdWithMetadata));

  auto object_detector_or = ObjectDetector::CreateFromOptions(
      options, absl::make_unique<MobileSsdQuantizedOpResolverMissingOps>());
  EXPECT_EQ(object_detector_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(object_detector_or.status().message(),
              HasSubstr("Encountered unresolved custom op"));
  EXPECT_THAT(object_detector_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kUnsupportedCustomOp))));
}

TEST_F(CreateFromOptionsTest, FailsWithTwoModelSources) {
  ObjectDetectorOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileSsdWithMetadata));
  options.mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileSsdWithMetadata));

  StatusOr<std::unique_ptr<ObjectDetector>> object_detector_or =
      ObjectDetector::CreateFromOptions(options);

  EXPECT_EQ(object_detector_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(object_detector_or.status().message(),
              HasSubstr("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 2."));
  EXPECT_THAT(object_detector_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  ObjectDetectorOptions options;

  StatusOr<std::unique_ptr<ObjectDetector>> object_detector_or =
      ObjectDetector::CreateFromOptions(options);

  EXPECT_EQ(object_detector_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(object_detector_or.status().message(),
              HasSubstr("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 0."));
  EXPECT_THAT(object_detector_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, FailsWithInvalidMaxResults) {
  ObjectDetectorOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileSsdWithMetadata));
  options.set_max_results(0);

  StatusOr<std::unique_ptr<ObjectDetector>> object_detector_or =
      ObjectDetector::CreateFromOptions(options);

  EXPECT_EQ(object_detector_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(object_detector_or.status().message(),
              HasSubstr("Invalid `max_results` option"));
  EXPECT_THAT(object_detector_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, FailsWithCombinedWhitelistAndBlacklist) {
  ObjectDetectorOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileSsdWithMetadata));
  options.add_class_name_whitelist("foo");
  options.add_class_name_blacklist("bar");

  StatusOr<std::unique_ptr<ObjectDetector>> object_detector_or =
      ObjectDetector::CreateFromOptions(options);

  EXPECT_EQ(object_detector_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(object_detector_or.status().message(),
              HasSubstr("mutually exclusive options"));
  EXPECT_THAT(object_detector_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, SucceedsWithNumberOfThreads) {
  ObjectDetectorOptions options;
  options.set_num_threads(4);
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileSsdWithMetadata));

  SUPPORT_ASSERT_OK(ObjectDetector::CreateFromOptions(options));
}

using NumThreadsTest = testing::TestWithParam<int>;

INSTANTIATE_TEST_SUITE_P(Default, NumThreadsTest, testing::Values(0, -2));

TEST_P(NumThreadsTest, FailsWithInvalidNumberOfThreads) {
  ObjectDetectorOptions options;
  options.set_num_threads(GetParam());
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileSsdWithMetadata));

  StatusOr<std::unique_ptr<ObjectDetector>> object_detector_or =
      ObjectDetector::CreateFromOptions(options);

  EXPECT_EQ(object_detector_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(object_detector_or.status().message(),
              HasSubstr("`num_threads` must be greater than "
                        "0 or equal to -1"));
  EXPECT_THAT(object_detector_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

class DetectTest : public tflite_shims::testing::Test {};

}  // namespace
}  // namespace vision
}  // namespace task
}  // namespace tflite
