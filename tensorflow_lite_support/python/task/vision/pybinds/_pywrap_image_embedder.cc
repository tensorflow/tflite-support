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
#include "tensorflow_lite_support/cc/task/vision/image_embedder.h"
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

PYBIND11_MODULE(_pywrap_image_embedder, m) {
  // python wrapper for C++ ImageEmbeder class which shouldn't be directly used
  // by the users.
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  py::class_<ImageEmbedder>(m, "ImageEmbedder")
      .def_static(
          "create_from_options",
          [](const PythonBaseOptions& base_options,
             const processor::EmbeddingOptions& embedding_options) {
            ImageEmbedderOptions options;
            if (base_options.has_file_content()) {
              options.mutable_model_file_with_metadata()->set_file_content(
                  base_options.file_content());
            }
            if (base_options.has_file_name()) {
              options.mutable_model_file_with_metadata()->set_file_name(
                  base_options.file_name());
            }

            options.set_num_threads(base_options.num_threads());
            if (base_options.use_coral()) {
              options.mutable_compute_settings()
                  ->mutable_tflite_settings()
                  ->set_delegate(tflite::proto::Delegate::EDGETPU_CORAL);
            }

            if (embedding_options.has_l2_normalize()) {
              options.set_l2_normalize(embedding_options.l2_normalize());
            }
            if (embedding_options.has_quantize()) {
              options.set_quantize(embedding_options.quantize());
            }
            return ImageEmbedder::CreateFromOptions(options);
          })
      .def("embed",
           [](ImageEmbedder& self, const ImageData& image_data)
               -> tflite::support::StatusOr<EmbeddingResult> {
             ASSIGN_OR_RETURN(std::unique_ptr<FrameBuffer> frame_buffer,
                              CreateFrameBufferFromImageData(image_data));
             return self.Embed(*frame_buffer);
           })
      .def("embed",
           [](ImageEmbedder& self, const ImageData& image_data,
              const BoundingBox& bounding_box)
               -> tflite::support::StatusOr<EmbeddingResult> {
             ASSIGN_OR_RETURN(std::unique_ptr<FrameBuffer> frame_buffer,
                              CreateFrameBufferFromImageData(image_data));
             return self.Embed(*frame_buffer, bounding_box);
           })
      .def("get_embedding_by_index", &ImageEmbedder::GetEmbeddingByIndex)
      .def("get_number_of_output_layers",
           &ImageEmbedder::GetNumberOfOutputLayers)
      .def("get_embedding_dimension", &ImageEmbedder::GetEmbeddingDimension)
      .def_static("cosine_similarity", &ImageEmbedder::CosineSimilarity);
}

}  // namespace vision
}  // namespace task
}  // namespace tflite
