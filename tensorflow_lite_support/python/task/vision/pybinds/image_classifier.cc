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

#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"

#include "pybind11/pybind11.h"
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

namespace tflite {
namespace task {
namespace vision {

namespace {
namespace py = ::pybind11;

static constexpr int kBuiltinOpResolver = 1;
static constexpr int kBuiltinRefOpResolver = 2;
static constexpr int kBuiltinOpResolverWithoutDefaultDelegates = 3;

std::unique_ptr<tflite::MutableOpResolver> get_resolver(int op_resolver_id) {
  std::unique_ptr<tflite::MutableOpResolver> resolver;
  switch (op_resolver_id) {
    case kBuiltinOpResolver:
      resolver = absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
      break;
    case kBuiltinRefOpResolver:
      resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinRefOpResolver>();
      break;
    case kBuiltinOpResolverWithoutDefaultDelegates:
      resolver = absl::make_unique<
          tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>();
      break;
    default:
      // This should not never happen because the eventual caller in
      // interpreter.py should have passed a valid id here.
      TFLITE_DCHECK(false);
      return nullptr;
  }
  return resolver;
}

}  // namespace

PYBIND11_MODULE(image_classifier, m) {
  // python wrapper for C++ ImageClassifier class which shouldn't be directly used
  // by the users.
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  py::class_<ImageClassifier>(m, "ImageClassifier")
      .def_static(
          "create_from_options",
          [](const ImageClassifierOptions& options, int op_resolver_id) {
            return ImageClassifier::CreateFromOptions(
                options, get_resolver(op_resolver_id));
          },
          py::arg("options"), py::arg("op_resolver_id") = kBuiltinOpResolver)
      .def("classify",
           [](ImageClassifier& self, const ImageData& image_data)
               -> tflite::support::StatusOr<ClassificationResult> {
             ASSIGN_OR_RETURN(std::unique_ptr<FrameBuffer> frame_buffer,
                              CreateFrameBufferFromImageData(image_data));
             return self.Classify(*frame_buffer);
           })
      .def("classify",
           [](ImageClassifier& self, const ImageData& image_data,
              const BoundingBox& bounding_box)
               -> tflite::support::StatusOr<ClassificationResult> {
             ASSIGN_OR_RETURN(std::unique_ptr<FrameBuffer> frame_buffer,
                              CreateFrameBufferFromImageData(image_data));
             return self.Classify(*frame_buffer, bounding_box);
           });
}

}  // namespace vision
}  // namespace task
}  // namespace tflite
