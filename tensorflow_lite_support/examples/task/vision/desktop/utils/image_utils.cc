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
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/io_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/task/core/external_file_handler.h"
#include "tensorflow_lite_support/cc/task/core/proto/external_file_proto_inc.h"

namespace tflite {
namespace support {
namespace task {
namespace vision {
namespace {

using ::tensorflow::Tensor;
using ::tensorflow::tstring;
using ::tensorflow::ops::DecodeBmp;
using ::tensorflow::ops::DecodeGif;
using ::tensorflow::ops::DecodeJpeg;
using ::tensorflow::ops::DecodePng;
using ::tensorflow::ops::Placeholder;
using ::tensorflow::ops::Squeeze;
using ::tflite::support::task::core::ExternalFile;
using ::tflite::support::task::core::ExternalFileHandler;

absl::Status ReadEntireFile(absl::string_view file_name, Tensor* output) {
  ExternalFile external_file;
  external_file.set_file_name(std::string(file_name));
  ASSIGN_OR_RETURN(std::unique_ptr<ExternalFileHandler> handler,
                   ExternalFileHandler::CreateFromExternalFile(&external_file));
  output->scalar<tstring>()() = tstring(handler->GetFileContent());
  return absl::OkStatus();
}

}  // namespace

// Core TensorFlow is used for convenience as it provides Ops able to decode
// various image formats. Any other image processing library like OpenCV or
// ImageMagick could be used as an alternative.
StatusOr<RgbImageData> DecodeImageFromFile(absl::string_view file_name) {
  // Read file_name into a tensor named input.
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  RETURN_IF_ERROR(ReadEntireFile(file_name, &input));

  auto root = tensorflow::Scope::NewRootScope();

  // Use a placeholder to read input data.
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  // Try to figure out what kind of file it is and decode it.
  const int wanted_channels = 3;  // for RGB output image
  tensorflow::Output image_reader;
  if (absl::EndsWithIgnoreCase(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("image_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (absl::EndsWithIgnoreCase(file_name, ".gif")) {
    // GIF decoder returns 4-D tensor, remove the first dim.
    image_reader = Squeeze(
        root.WithOpName("image_reader"),
        DecodeGif(root.WithOpName("image_reader_before_squeeze"), file_reader));
  } else if (absl::EndsWithIgnoreCase(file_name, ".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("image_reader"), file_reader);
  } else if (absl::EndsWithIgnoreCase(file_name, ".jpeg") ||
             absl::EndsWithIgnoreCase(file_name, ".jpg")) {
    image_reader = DecodeJpeg(root.WithOpName("image_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  } else {
    return absl::UnimplementedError(
        "Only .png, .gif, .bmp and .jpg (or .jpeg) images are supported");
  }

  // This runs the GraphDef network definition constructed above, and returns
  // the results in an output tensor.
  tensorflow::GraphDef graph;
  if (!root.ToGraphDef(&graph).ok()) {
    return absl::InternalError(
        "Initialization error while decoding input image.");
  }

  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  if (!session->Create(graph).ok()) {
    return absl::InternalError(
        "Initialization error while decoding input image.");
  }

  std::vector<std::pair<std::string, Tensor>> inputs = {
      {"input", input},
  };
  std::vector<Tensor> output_tensors;

  tensorflow::Status status =
      session->Run({inputs}, /*output_tensor_names=*/{"image_reader"},
                   /*target_node_names=*/{}, &output_tensors);
  if (!status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "An internal error occurred while decoding input image: %s",
        status.error_message()));
  }

  // A single output tensor with shape `[height, width, channels]` where
  // `channels=3` is expected.
  if (output_tensors.size() != 1 ||
      output_tensors[0].dtype() != tensorflow::DT_UINT8 ||
      output_tensors[0].dims() != 3 ||
      output_tensors[0].shape().dim_size(2) != 3) {
    return absl::InternalError("Unexpected output after decoding input image.");
  }

  RgbImageData image_data;
  size_t total_bytes = output_tensors[0].NumElements() * sizeof(uint8);
  image_data.pixel_data = static_cast<uint8*>(malloc(total_bytes));
  memcpy(image_data.pixel_data, output_tensors[0].flat<uint8>().data(),
         total_bytes);
  image_data.height = output_tensors[0].shape().dim_size(0);
  image_data.width = output_tensors[0].shape().dim_size(1);

  return image_data;
}

absl::Status EncodeRgbImageToPngFile(const RgbImageData& image_data,
                                     absl::string_view image_path) {
  // Sanity check inputs.
  if (image_data.width <= 0 || image_data.height <= 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected positive image dimensions, found %d x %d.",
                        image_data.width, image_data.height));
  }
  if (image_data.pixel_data == nullptr) {
    return absl::InvalidArgumentError(
        "Expected pixel data to be set, found nullptr.");
  }

  // Prepare input tensor.
  Tensor input(tensorflow::DataType::DT_UINT8,
               {image_data.height, image_data.width, 3});
  auto data = input.flat<uint8>().data();
  for (int i = 0; i < input.NumElements(); ++i) {
    *data = image_data.pixel_data[i];
    ++data;
  }
  std::vector<std::pair<std::string, Tensor>> inputs = {
      {"input", input},
  };
  // Convert path to tensorflow string.
  const tensorflow::string output_path(image_path);

  // Build graph.
  auto root = tensorflow::Scope::NewRootScope();
  tensorflow::Output placeholder =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_UINT8);
  tensorflow::Output png_encoder =
      tensorflow::ops::EncodePng(root.WithOpName("pngencoder"), placeholder);
  tensorflow::ops::WriteFile file_writer = tensorflow::ops::WriteFile(
      root.WithOpName("output"), output_path, png_encoder);
  tensorflow::GraphDef graph;
  if (!root.ToGraphDef(&graph).ok()) {
    return absl::InternalError(
        "Initialization error while encoding image to PNG.");
  }

  // Create session and run graph.
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
  if (!session->Create(graph).ok()) {
    return absl::InternalError(
        "Initialization error while encoding image to PNG.");
  }
  tensorflow::Status status =
      session->Run({inputs}, /*output_tensor_names=*/{},
                   /*target_node_names=*/{"output"}, /*outputs=*/nullptr);
  if (!status.ok()) {
    return absl::InternalError(absl::StrFormat(
        "An internal error occurred while encoding image to PNG: %s",
        status.error_message()));
  }

  return absl::OkStatus();
}

void RgbImageDataFree(RgbImageData* image) { free(image->pixel_data); }

}  // namespace vision
}  // namespace task
}  // namespace support
}  // namespace tflite
