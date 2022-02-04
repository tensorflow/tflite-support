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

#include "tensorflow_lite_support/cc/task/vision/image_embedder.h"

#include "pybind11/pybind11.h"
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

namespace tflite {
namespace task {
namespace vision {

namespace {
namespace py = ::pybind11;
}  // namespace

PYBIND11_MODULE(_pywrap_image_embedder, m) {
  // python wrapper for C++ ImageEmbeder class which shouldn't be directly used
  // by the users.
  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  py::class_<ImageEmbedder>(m, "ImageEmbedder")
      .def_static(
          "create_from_options",
          [](const ImageEmbedderOptions& options) {
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
