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

#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"

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
#include "tensorflow_lite_support/cc/task/vision/proto/bounding_box_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_classifier_options_proto_inc.h"
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
using ::testing::ElementsAreArray;
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
// Float model.
constexpr char kMobileNetFloatWithMetadata[] = "mobilenet_v2_1.0_224.tflite";
// Quantized model.
constexpr char kMobileNetQuantizedWithMetadata[] =
    "mobilenet_v1_0.25_224_quant.tflite";
// Hello world flowers classifier supporting 5 classes (quantized model).
constexpr char kAutoMLModelWithMetadata[] = "automl_labeler_model.tflite";

StatusOr<ImageData> LoadImage(std::string image_name) {
  return DecodeImageFromFile(JoinPath("./" /*test src dir*/,
                                      kTestDataDirectory, image_name));
}

class MobileNetQuantizedOpResolver : public ::tflite::MutableOpResolver {
 public:
  MobileNetQuantizedOpResolver() {
    AddBuiltin(::tflite::BuiltinOperator_AVERAGE_POOL_2D,
               ::tflite::ops::builtin::Register_AVERAGE_POOL_2D());
    AddBuiltin(::tflite::BuiltinOperator_CONV_2D,
               ::tflite::ops::builtin::Register_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
               ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
    AddBuiltin(::tflite::BuiltinOperator_RESHAPE,
               ::tflite::ops::builtin::Register_RESHAPE());
    AddBuiltin(::tflite::BuiltinOperator_SOFTMAX,
               ::tflite::ops::builtin::Register_SOFTMAX());
  }

  MobileNetQuantizedOpResolver(const MobileNetQuantizedOpResolver& r) = delete;
};

class CreateFromOptionsTest : public tflite_shims::testing::Test {};

TEST_F(CreateFromOptionsTest, SucceedsWithSelectiveOpResolver) {
  ImageClassifierOptions options;
  options.set_max_results(3);
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata));

  SUPPORT_ASSERT_OK(ImageClassifier::CreateFromOptions(
      options, absl::make_unique<MobileNetQuantizedOpResolver>()));
}

class MobileNetQuantizedOpResolverMissingOps
    : public ::tflite::MutableOpResolver {
 public:
  MobileNetQuantizedOpResolverMissingOps() {
    AddBuiltin(::tflite::BuiltinOperator_SOFTMAX,
               ::tflite::ops::builtin::Register_SOFTMAX());
  }

  MobileNetQuantizedOpResolverMissingOps(
      const MobileNetQuantizedOpResolverMissingOps& r) = delete;
};

TEST_F(CreateFromOptionsTest, FailsWithSelectiveOpResolverMissingOps) {
  ImageClassifierOptions options;
  options.set_max_results(3);
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata));

  auto image_classifier_or = ImageClassifier::CreateFromOptions(
      options, absl::make_unique<MobileNetQuantizedOpResolverMissingOps>());
  EXPECT_EQ(image_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier_or.status().message(),
              HasSubstr("Didn't find op for builtin opcode"));
  EXPECT_THAT(image_classifier_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kUnsupportedBuiltinOp))));
}

TEST_F(CreateFromOptionsTest, FailsWithTwoModelSources) {
  ImageClassifierOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata));
  options.mutable_base_options()->mutable_model_file()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetFloatWithMetadata));

  StatusOr<std::unique_ptr<ImageClassifier>> image_classifier_or =
      ImageClassifier::CreateFromOptions(options);

  EXPECT_EQ(image_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier_or.status().message(),
              HasSubstr("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 2."));
  EXPECT_THAT(image_classifier_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, FailsWithMissingModel) {
  ImageClassifierOptions options;

  StatusOr<std::unique_ptr<ImageClassifier>> image_classifier_or =
      ImageClassifier::CreateFromOptions(options);

  EXPECT_EQ(image_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier_or.status().message(),
              HasSubstr("Expected exactly one of `base_options.model_file` or "
                        "`model_file_with_metadata` to be provided, found 0."));
  EXPECT_THAT(image_classifier_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, FailsWithInvalidMaxResults) {
  ImageClassifierOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata));
  options.set_max_results(0);

  StatusOr<std::unique_ptr<ImageClassifier>> image_classifier_or =
      ImageClassifier::CreateFromOptions(options);

  EXPECT_EQ(image_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier_or.status().message(),
              HasSubstr("Invalid `max_results` option"));
  EXPECT_THAT(image_classifier_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, FailsWithCombinedWhitelistAndBlacklist) {
  ImageClassifierOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetQuantizedWithMetadata));
  options.add_class_name_whitelist("foo");
  options.add_class_name_blacklist("bar");

  StatusOr<std::unique_ptr<ImageClassifier>> image_classifier_or =
      ImageClassifier::CreateFromOptions(options);

  EXPECT_EQ(image_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier_or.status().message(),
              HasSubstr("mutually exclusive options"));
  EXPECT_THAT(image_classifier_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

TEST_F(CreateFromOptionsTest, SucceedsWithNumberOfThreads) {
  ImageClassifierOptions options;
  options.set_num_threads(4);
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetFloatWithMetadata));

  SUPPORT_ASSERT_OK(ImageClassifier::CreateFromOptions(options));
}

using NumThreadsTest = testing::TestWithParam<int>;

INSTANTIATE_TEST_SUITE_P(Default, NumThreadsTest, testing::Values(0, -2));

TEST_P(NumThreadsTest, FailsWithInvalidNumberOfThreads) {
  ImageClassifierOptions options;
  options.set_num_threads(GetParam());
  options.mutable_model_file_with_metadata()->set_file_name(
      JoinPath("./" /*test src dir*/, kTestDataDirectory,
               kMobileNetFloatWithMetadata));

  StatusOr<std::unique_ptr<ImageClassifier>> image_classifier_or =
      ImageClassifier::CreateFromOptions(options);

  EXPECT_EQ(image_classifier_or.status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(image_classifier_or.status().message(),
              HasSubstr("`num_threads` must be greater than "
                        "0 or equal to -1"));
  EXPECT_THAT(image_classifier_or.status().GetPayload(kTfLiteSupportPayload),
              Optional(absl::Cord(
                  absl::StrCat(TfLiteSupportStatus::kInvalidArgumentError))));
}

}  // namespace
}  // namespace vision
}  // namespace task
}  // namespace tflite
