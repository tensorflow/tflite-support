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

#include "tensorflow_lite_support/cc/task/vision/image_segmenter.h"

#include <memory>

#include "external/com_google_absl/absl/flags/flag.h"
#include "external/com_google_absl/absl/status/status.h"
#include "external/com_google_absl/absl/strings/cord.h"
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
#include "tensorflow_lite_support/cc/task/vision/proto/image_segmenter_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/segmentations_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_utils.h"
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
using ::tflite::task::core::PopulateTensor;
using ::tflite::task::core::TaskAPIFactory;
using ::tflite::task::core::TfLiteEngine;

constexpr char kTestDataDirectory[] =
    "/tensorflow_lite_support/cc/test/testdata/task/"
    "vision/";
constexpr char kDeepLabV3[] = "deeplabv3.tflite";

// All results returned by DeepLabV3 are expected to contain these in addition
// to the segmentation masks.
constexpr char kDeepLabV3PartialResult[] =
    R"(width: 257
       height: 257
       colored_labels { r: 0 g: 0 b: 0 class_name: "background" }
       colored_labels { r: 128 g: 0 b: 0 class_name: "aeroplane" }
       colored_labels { r: 0 g: 128 b: 0 class_name: "bicycle" }
       colored_labels { r: 128 g: 128 b: 0 class_name: "bird" }
       colored_labels { r: 0 g: 0 b: 128 class_name: "boat" }
       colored_labels { r: 128 g: 0 b: 128 class_name: "bottle" }
       colored_labels { r: 0 g: 128 b: 128 class_name: "bus" }
       colored_labels { r: 128 g: 128 b: 128 class_name: "car" }
       colored_labels { r: 64 g: 0 b: 0 class_name: "cat" }
       colored_labels { r: 192 g: 0 b: 0 class_name: "chair" }
       colored_labels { r: 64 g: 128 b: 0 class_name: "cow" }
       colored_labels { r: 192 g: 128 b: 0 class_name: "dining table" }
       colored_labels { r: 64 g: 0 b: 128 class_name: "dog" }
       colored_labels { r: 192 g: 0 b: 128 class_name: "horse" }
       colored_labels { r: 64 g: 128 b: 128 class_name: "motorbike" }
       colored_labels { r: 192 g: 128 b: 128 class_name: "person" }
       colored_labels { r: 0 g: 64 b: 0 class_name: "potted plant" }
       colored_labels { r: 128 g: 64 b: 0 class_name: "sheep" }
       colored_labels { r: 0 g: 192 b: 0 class_name: "sofa" }
       colored_labels { r: 128 g: 192 b: 0 class_name: "train" }
       colored_labels { r: 0 g: 64 b: 128 class_name: "tv" })";

// The maximum fraction of pixels in the candidate mask that can have a
// different class than the golden mask for the test to pass.
constexpr float kGoldenMaskTolerance = 1e-2;
// Magnification factor used when creating the golden category masks to make
// them more human-friendly. Each pixel in the golden masks has its value
// multiplied by this factor, i.e. a value of 10 means class index 1, a value of
// 20 means class index 2, etc.
constexpr int kGoldenMaskMagnificationFactor = 10;

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

class DeepLabOpResolver : public ::tflite::MutableOpResolver {
 public:
  DeepLabOpResolver() {
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
    AddBuiltin(::tflite::BuiltinOperator_AVERAGE_POOL_2D,
               ::tflite::ops::builtin::Register_AVERAGE_POOL_2D());
    AddBuiltin(::tflite::BuiltinOperator_CONCATENATION,
               ::tflite::ops::builtin::Register_CONCATENATION());
    AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
               ::tflite::ops::builtin::Register_CONV_2D());
    // DeepLab uses different versions of DEPTHWISE_CONV_2D.
    AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D(),
               /*min_version=*/1, /*max_version=*/2);
    AddBuiltin(::tflite::BuiltinOperator_RESIZE_BILINEAR,
               ::tflite::ops::builtin::Register_RESIZE_BILINEAR());
  }

  DeepLabOpResolver(const DeepLabOpResolver& r) = delete;
};

class CreateFromOptionsTest : public tflite_shims::testing::Test {};

TEST_F(CreateFromOptionsTest, SucceedsWithSelectiveOpResolver) {
  ImageSegmenterOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(JoinPath(
      "./" /*test src dir*/, kTestDataDirectory, kDeepLabV3));

  SUPPORT_ASSERT_OK(ImageSegmenter::CreateFromOptions(
      options, absl::make_unique<DeepLabOpResolver>()));
}

class DeepLabOpResolverMissingOps : public ::tflite::MutableOpResolver {
 public:
  DeepLabOpResolverMissingOps() {
    AddBuiltin(::tflite::BuiltinOperator_ADD,
               ::tflite::ops::builtin::Register_ADD());
  }

  DeepLabOpResolverMissingOps(const DeepLabOpResolverMissingOps& r) = delete;
};

TEST_F(CreateFromOptionsTest, FailsWithSelectiveOpResolverMissingOps) {
  ImageSegmenterOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(JoinPath(
      "./" /*test src dir*/, kTestDataDirectory, kDeepLabV3));

  auto image_segmenter_or = ImageSegmenter::CreateFromOptions(
      options, absl::make_unique<DeepLabOpResolverMissingOps>());

  EXPECT_EQ(image_segmenter_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_segmenter_or.status().message(),
              HasSubstr("Didn't find op for builtin opcode"));
  EXPECT_THAT(image_segmenter_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kUnsupportedBuiltinOp))));
}

TEST_F(CreateFromOptionsTest, FailsWithTwoModelSources) {
  ImageSegmenterOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(JoinPath(
      "./" /*test src dir*/, kTestDataDirectory, kDeepLabV3));
  options.mutable_base_options()->mutable_model_file()->set_file_name(JoinPath(
      "./" /*test src dir*/, kTestDataDirectory, kDeepLabV3));

  StatusOr<std::unique_ptr<ImageSegmenter>> image_segmenter_or =
      ImageSegmenter::CreateFromOptions(options);

  EXPECT_EQ(image_segmenter_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_segmenter_or.status().message(),
              HasSubstr("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 2."));
  EXPECT_THAT(image_segmenter_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  ImageSegmenterOptions options;

  auto image_segmenter_or = ImageSegmenter::CreateFromOptions(options);

  EXPECT_EQ(image_segmenter_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_segmenter_or.status().message(),
              HasSubstr("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 0."));
  EXPECT_THAT(image_segmenter_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, FailsWithUnspecifiedOutputType) {
  ImageSegmenterOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(JoinPath(
      "./" /*test src dir*/, kTestDataDirectory, kDeepLabV3));
  options.set_output_type(ImageSegmenterOptions::UNSPECIFIED);

  auto image_segmenter_or = ImageSegmenter::CreateFromOptions(options);

  EXPECT_EQ(image_segmenter_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_segmenter_or.status().message(),
              HasSubstr("`output_type` must not be UNSPECIFIED"));
  EXPECT_THAT(image_segmenter_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, SucceedsWithNumberOfThreads) {
  ImageSegmenterOptions options;
  options.set_num_threads(4);
  options.mutable_model_file_with_metadata()->set_file_name(JoinPath(
      "./" /*test src dir*/, kTestDataDirectory, kDeepLabV3));

  SUPPORT_ASSERT_OK(ImageSegmenter::CreateFromOptions(options));
}

using NumThreadsTest = testing::TestWithParam<int>;

INSTANTIATE_TEST_SUITE_P(Default, NumThreadsTest, testing::Values(0, -2));

TEST_P(NumThreadsTest, FailsWithInvalidNumberOfThreads) {
  ImageSegmenterOptions options;
  options.set_num_threads(GetParam());
  options.mutable_model_file_with_metadata()->set_file_name(JoinPath(
      "./" /*test src dir*/, kTestDataDirectory, kDeepLabV3));

  StatusOr<std::unique_ptr<ImageSegmenter>> image_segmenter_or =
      ImageSegmenter::CreateFromOptions(options);

  EXPECT_EQ(image_segmenter_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_segmenter_or.status().message(),
              HasSubstr("`num_threads` must be greater than "
                        "0 or equal to -1"));
  EXPECT_THAT(image_segmenter_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

}  // namespace
}  // namespace vision
}  // namespace task
}  // namespace tflite
