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
#ifndef TENSORFLOW_LITE_SUPPORT_EXAMPLES_TASK_VISION_DESKTOP_UTILS_IMAGE_UTILS_H_
#define TENSORFLOW_LITE_SUPPORT_EXAMPLES_TASK_VISION_DESKTOP_UTILS_IMAGE_UTILS_H_

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow_lite_support/cc/port/integral_types.h"
#include "tensorflow_lite_support/cc/port/statusor.h"

namespace tflite {
namespace support {
namespace task {
namespace vision {

// Interleaved RGB image with pixels stored as a row-major flattened array.
struct RgbImageData {
  uint8* pixel_data;
  int width;
  int height;
};

// Decodes image file and returns the corresponding RGB image if no error
// occurred. Supported formats are JPEG, PNG, GIF and BMP. If decoding
// succeeded, the caller must manage deletion of the underlying pixel data using
// `RgbImageDataFree`.
StatusOr<RgbImageData> DecodeImageFromFile(absl::string_view file_name);

// Encodes the image provided as an RgbImageData as lossless PNG to the provided
// path.
absl::Status EncodeRgbImageToPngFile(const RgbImageData& image_data,
                                     absl::string_view image_path);

// Releases image pixel data memory.
void RgbImageDataFree(RgbImageData* image);

}  // namespace vision
}  // namespace task
}  // namespace support
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_EXAMPLES_TASK_VISION_DESKTOP_UTILS_IMAGE_UTILS_H_
