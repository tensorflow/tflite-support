/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow_lite_support/cc/task/text/text_embedder.h"

#include "pybind11/pybind11.h"
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf

namespace tflite {
namespace task {
namespace text {

PYBIND11_MODULE(_pywrap_text_embedder, m) {
  // python wrapper for C++ TextEmbeder class which shouldn't be directly used
  // by the users.

  pybind11::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  pybind11::class_<TextEmbedder>(m, "TextEmbedder")
      .def_static("create_from_options",
                  [](const TextEmbedderOptions& options) {
                    return TextEmbedder::CreateFromOptions(options);
                  })
      .def("embed", &TextEmbedder::Embed)
      .def("get_embedding_dimension", &TextEmbedder::GetEmbeddingDimension)
      .def("get_number_of_output_layers",
           &TextEmbedder::GetNumberOfOutputLayers)
      .def_static("cosine_similarity", &TextEmbedder::CosineSimilarity);
}

}  // namespace text
}  // namespace task
}  // namespace tflite
