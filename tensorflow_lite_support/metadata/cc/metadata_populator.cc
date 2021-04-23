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

#include "tensorflow_lite_support/metadata/cc/metadata_populator.h"

#include <cstdlib>
#include <cstring>
#include <functional>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "contrib/minizip/ioapi.h"
#include "contrib/minizip/zip.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/metadata/cc/utils/zip_mem_file.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace metadata {

namespace {
constexpr absl::string_view kMetadataBufferName = "TFLITE_METADATA";

using ::absl::StatusCode;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::TfLiteSupportStatus;

}  // namespace

ModelMetadataPopulator::ModelMetadataPopulator(const tflite::Model& model) {
  model.UnPackTo(&model_t_);
}

/* static */
tflite::support::StatusOr<std::unique_ptr<ModelMetadataPopulator>>
ModelMetadataPopulator::CreateFromModelBuffer(const char* buffer_data,
                                              size_t buffer_size) {
  // Rely on the simplest, base flatbuffers verifier. Here is not the place to
  // e.g. use an OpResolver: we just want to make sure the buffer is valid to
  // access the metadata.
  flatbuffers::Verifier verifier = flatbuffers::Verifier(
      reinterpret_cast<const uint8_t*>(buffer_data), buffer_size);
  if (!tflite::VerifyModelBuffer(verifier)) {
    return CreateStatusWithPayload(
        StatusCode::kInvalidArgument,
        "The model is not a valid FlatBuffer buffer.",
        TfLiteSupportStatus::kInvalidFlatBufferError);
  }
  // Use absl::WrapUnique() to call private constructor:
  // https://abseil.io/tips/126.
  return absl::WrapUnique(
      new ModelMetadataPopulator(*tflite::GetModel(buffer_data)));
}

void ModelMetadataPopulator::LoadMetadata(const char* metadata_buffer_data,
                                          size_t metadata_buffer_size) {
  // Pack the model metadata in a buffer.
  auto model_metadata_buffer = std::make_unique<tflite::BufferT>();
  model_metadata_buffer->data = {metadata_buffer_data,
                                 metadata_buffer_data + metadata_buffer_size};
  // Check if the model already has metadata. If so, just override the buffer
  // and exit.
  for (const auto& metadata_t : model_t_.metadata) {
    if (metadata_t->name == kMetadataBufferName) {
      model_t_.buffers[metadata_t->buffer] = std::move(model_metadata_buffer);
      return;
    }
  }
  // Model doesn't already have metadata: add metadata buffer and pointer to the
  // buffer in the model metadata section.
  model_t_.buffers.push_back(std::move(model_metadata_buffer));
  auto metadata_t = std::make_unique<tflite::MetadataT>();
  metadata_t->name = kMetadataBufferName;
  metadata_t->buffer = model_t_.buffers.size() - 1;
  model_t_.metadata.push_back(std::move(metadata_t));
}

void ModelMetadataPopulator::LoadAssociatedFiles(
    const absl::flat_hash_map<std::string, std::string>& associated_files) {
  associated_files_ = associated_files;
}

tflite::support::StatusOr<std::string>
ModelMetadataPopulator::AppendAssociatedFiles(const char* model_buffer_data,
                                              size_t model_buffer_size) {
  // Create in-memory zip file.
  ZipMemFile mem_file = ZipMemFile(model_buffer_data, model_buffer_size);
  // Open zip.
  zipFile zf = zipOpen2(/*pathname=*/nullptr, APPEND_STATUS_CREATEAFTER,
                        /*globalcomment=*/nullptr, &mem_file.GetFileFuncDef());
  if (zf == nullptr) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Unable to open zip archive",
        TfLiteSupportStatus::kMetadataAssociatedFileZipError);
  }
  // Write associated files.
  for (const auto& [name, contents] : associated_files_) {
    if ((zipOpenNewFileInZip(zf, name.c_str(),
                             /*zipfi=*/nullptr,
                             /*extrafield_local=*/nullptr,
                             /*size_extrafield_local=*/0,
                             /*extrafield_global=*/nullptr,
                             /*size_extrafield_global=*/0,
                             /*comment=*/nullptr,
                             /*method=*/0,
                             /*level=*/Z_DEFAULT_COMPRESSION) != ZIP_OK) ||
        (zipWriteInFileInZip(zf, contents.data(), contents.length()) !=
         ZIP_OK) ||
        (zipCloseFileInZip(zf) != ZIP_OK)) {
      return CreateStatusWithPayload(
          StatusCode::kUnknown, "Unable to write file to zip archive",
          TfLiteSupportStatus::kMetadataAssociatedFileZipError);
    }
  }
  // Close zip.
  if (zipClose(zf, /*global_comment=*/nullptr) != ZIP_OK) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Unable to close zip archive",
        TfLiteSupportStatus::kMetadataAssociatedFileZipError);
  }
  // Return as a string.
  return std::string(mem_file.GetFileContent());
}

tflite::support::StatusOr<std::string> ModelMetadataPopulator::Populate() {
  // Build model.
  flatbuffers::FlatBufferBuilder model_fbb;
  model_fbb.Finish(tflite::Model::Pack(model_fbb, &model_t_),
                   tflite::ModelIdentifier());
  return AppendAssociatedFiles(
      reinterpret_cast<char*>(model_fbb.GetBufferPointer()),
      model_fbb.GetSize());
}

}  // namespace metadata
}  // namespace tflite
