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

#include "tensorflow_lite_support/cc/task/vision/landmark_detector.h"

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/integral_types.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/task/core/tflite_engine.h"
#include "tensorflow_lite_support/cc/task/vision/proto/landmarks_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_utils.h"
#include "tensorflow_lite_support/metadata/cc/metadata_extractor.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace task {
namespace vision {

namespace {

using ::absl::StatusCode;
using ::tflite::metadata::ModelMetadataExtractor;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::core::AssertAndReturnTypedTensor;
using ::tflite::task::core::TaskAPIFactory;
using ::tflite::task::core::TfLiteEngine;



}  // namespace

// Number of Keypoints
int num_keypoints = 17;

/* static */
StatusOr<std::unique_ptr<LandmarkDetector>> LandmarkDetector::CreateFromOptions(
    const LandmarkDetectorOptions& options) {
  RETURN_IF_ERROR(SanityCheckOptions(options));

  // Copy options to ensure the ExternalFile outlives the constructed object.
  auto options_copy = absl::make_unique<LandmarkDetectorOptions>(options);

  std::unique_ptr<LandmarkDetector> landmark_detector;
  if (options_copy->base_options().has_model_file()) {
    ASSIGN_OR_RETURN(landmark_detector,
                     TaskAPIFactory::CreateFromBaseOptions<LandmarkDetector>(
                         &options_copy->base_options()));
  } else {
    // Should never happen because of SanityCheckOptions.
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Expected exactly one `base_options.model_file` "
                        "to be provided, found 0."),
        TfLiteSupportStatus::kInvalidArgumentError);
  }

  RETURN_IF_ERROR(landmark_detector->Init(std::move(options_copy)));

  return landmark_detector;
}

/* static */
absl::Status LandmarkDetector::SanityCheckOptions(
    const LandmarkDetectorOptions& options) {
  // Nothing to check
  return absl::OkStatus();
}

absl::Status LandmarkDetector::Init(
    std::unique_ptr<LandmarkDetectorOptions> options) {
  // Set options.
  options_ = std::move(options);

  // Perform pre-initialization actions (by default, sets the process engine for
  // image pre-processing to kLibyuv as a sane default).
  RETURN_IF_ERROR(PreInit());

  // Sanity check and set inputs and outputs.
  RETURN_IF_ERROR(CheckAndSetInputs());

  return absl::OkStatus();
}

absl::Status LandmarkDetector::PreInit() {
  SetProcessEngine(FrameBufferUtils::ProcessEngine::kLibyuv);
  return absl::OkStatus();
}

StatusOr<LandmarkResult> LandmarkDetector::Detect(
    const FrameBuffer& frame_buffer) {
  BoundingBox roi;
  roi.set_width(frame_buffer.dimension().width);
  roi.set_height(frame_buffer.dimension().height);
  return Detect(frame_buffer, roi);
}

StatusOr<LandmarkResult> LandmarkDetector::Detect(
    const FrameBuffer& frame_buffer, const BoundingBox& roi) {
  return InferWithFallback(frame_buffer, roi);
}

StatusOr<LandmarkResult> LandmarkDetector::Postprocess(
    const std::vector<const TfLiteTensor*>& output_tensors,
    const FrameBuffer& /*frame_buffer*/, const BoundingBox& /*roi*/) {
  if (output_tensors.size() != 1) {
    return CreateStatusWithPayload(
        StatusCode::kInternal,
        absl::StrFormat("Expected 1 output tensors, found %d",
                        output_tensors.size()));
  }

  const TfLiteTensor* output_tensor = output_tensors[0];

  const float* outputs = AssertAndReturnTypedTensor<float>(output_tensor);
	
  LandmarkResult result;

	for(int i =0 ; i<num_keypoints ; ++i){

    Landmark* landmarks = result.add_landmarks();

		landmarks->set_key_y(outputs[3*i+0]);
		landmarks->set_key_x(outputs[3*i+1]);
		landmarks->set_score(outputs[3*i+2]);

  }
  
  return result;
}


}  // namespace vision
}  // namespace task
}  // namespace tflite