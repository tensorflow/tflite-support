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

#include "external/com_google_absl/absl/strings/str_format.h"
#include "external/com_google_absl/absl/strings/string_view.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"

namespace tflite {
namespace task {
namespace vision {

namespace {

using ::absl::StatusCode;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::StatusOr;
using ::tflite::support::TfLiteSupportStatus;
using ::tflite::task::core::AssertAndReturnTypedTensor;
using ::tflite::task::core::TaskAPIFactory;
using ::tflite::task::core::TfLiteEngine;
using ::tflite::task::vision::FrameBuffer;
}  // namespace

/* static */
StatusOr<std::unique_ptr<ImageTransformer>> ImageTransformer::CreateFromOptions(
    const ImageTransformerOptions& options,
    std::unique_ptr<tflite::OpResolver> resolver) {
  RETURN_IF_ERROR(SanityCheckOptions(options));

  // Copy options to ensure the ExternalFile outlives the constructed object.
  auto options_copy = absl::make_unique<ImageTransformerOptions>(options);

  std::unique_ptr<ImageTransformer> image_transformer;

  ASSIGN_OR_RETURN(image_transformer,
                   TaskAPIFactory::CreateFromBaseOptions<ImageTransformer>(
                       &options_copy->base_options(), std::move(resolver)));

  RETURN_IF_ERROR(image_transformer->Init(std::move(options_copy)));
  return image_transformer;
}

/* static */
absl::Status ImageTransformer::SanityCheckOptions(
    const ImageTransformerOptions& options) {
  // Nothing to do.
  return absl::OkStatus();
}

absl::Status ImageTransformer::Init(
    std::unique_ptr<ImageTransformerOptions> options) {
  // Set options.
  options_ = std::move(options);

  // Perform pre-initialization actions (by default, sets the process engine for
  // image pre-processing to kLibyuv as a sane default).
  RETURN_IF_ERROR(PreInit());

  // Sanity check and set inputs and outputs.
  RETURN_IF_ERROR(CheckAndSetInputs());
  RETURN_IF_ERROR(CheckAndSetOutputs());

  RETURN_IF_ERROR(PostInit());

  ASSIGN_OR_RETURN(postprocessor_, processor::ImagePostprocessor::Create(
                                       GetTfLiteEngine(), {0}, {0}));

  return absl::OkStatus();
}

absl::Status ImageTransformer::PreInit() {
  SetProcessEngine(FrameBufferUtils::ProcessEngine::kLibyuv);
  return absl::OkStatus();
}

absl::Status ImageTransformer::PostInit() {
  // Nothing to do.
  return absl::OkStatus();
}

absl::Status ImageTransformer::CheckAndSetOutputs() {
  // Check the number of output tensors.
  if (TfLiteEngine::OutputCount(GetTfLiteEngine()->interpreter()) != 1) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        absl::StrFormat("Image segmentation models are expected to have only 1 "
                        "output, found %d",
                        TfLiteEngine::OutputCount(interpreter)),
        TfLiteSupportStatus::kInvalidNumOutputTensorsError);
  }
  return absl::OkStatus();
}

StatusOr<FrameBuffer> ImageTransformer::Transform(
    const FrameBuffer& frame_buffer) {
  BoundingBox roi;
  roi.set_width(frame_buffer.dimension().width);
  roi.set_height(frame_buffer.dimension().height);
  return Transform(frame_buffer, roi);
}

StatusOr<FrameBuffer> ImageTransformer::Transform(
    const FrameBuffer& frame_buffer, const BoundingBox& roi) {
  return InferWithFallback(frame_buffer, roi);
}

StatusOr<FrameBuffer> ImageTransformer::Postprocess(
    const std::vector<const TfLiteTensor*>& /*output_tensors*/,
    const FrameBuffer& /*frame_buffer*/, const BoundingBox& /*roi*/) {
  ASSIGN_OR_RETURN(auto postprocessed_output, postprocessor_->Postprocess());
  return postprocessed_output;
}
}  // namespace vision
}  // namespace task
}  // namespace tflite
