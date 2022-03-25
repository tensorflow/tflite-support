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

#include "pybind11/pybind11.h"
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/processor/proto/classification_options.pb.h"
#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"
#include "tensorflow_lite_support/python/task/core/pybinds/task_utils.h"

namespace tflite {
namespace task {
namespace vision {

namespace {
namespace py = ::pybind11;
using PythonBaseOptions = ::tflite::python::task::core::BaseOptions;
using CppBaseOptions = ::tflite::task::core::BaseOptions;
}  // namespace

PYBIND11_MODULE(_pywrap_image_classifier, m) {
  // python wrapper for C++ ImageClassifier class which shouldn't be directly
  // used by the users.
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  py::class_<ImageClassifier>(m, "ImageClassifier")
      .def_static(
          "create_from_options",
          [](const PythonBaseOptions& base_options,
             const processor::ClassificationOptions& classification_options) {
            ImageClassifierOptions options;
            auto cpp_base_options =
                core::convert_to_cpp_base_options(base_options);
            options.set_allocated_base_options(cpp_base_options.release());

            if (classification_options.has_display_names_locale()) {
              options.set_display_names_locale(
                  classification_options.display_names_locale());
            }
            if (classification_options.has_max_results()) {
              options.set_max_results(classification_options.max_results());
            }
            if (classification_options.has_score_threshold()) {
              options.set_score_threshold(
                  classification_options.score_threshold());
            }
            options.mutable_class_name_whitelist()->CopyFrom(
                classification_options.class_name_allowlist());
            options.mutable_class_name_blacklist()->CopyFrom(
                classification_options.class_name_denylist());

            return ImageClassifier::CreateFromOptions(options);
          })
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
