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

#include <functional>

#include "absl/strings/str_format.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "miniz_zip.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow_lite_support/cc/common.h"
#include "tensorflow_lite_support/cc/port/status_macros.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace metadata {

namespace {
constexpr char kMetadataBufferName[] = "TFLITE_METADATA";

using ::absl::StatusCode;
using ::tflite::support::CreateStatusWithPayload;
using ::tflite::support::TfLiteSupportStatus;

// Helper class that takes a callback function, and invokes it in its
// destructor.
class SimpleCleanUp {
 public:
  explicit SimpleCleanUp(std::function<void()> callback)
      : callback_(std::move(callback)) {}

  ~SimpleCleanUp() {
    if (callback_ != nullptr) callback_();
  }

  // Use `std::move(simple_cleanup).Cancel()` to prevent the callback from ever
  // executing at all. Once a SimpleCleanUp object has been `std::move(...)`-ed,
  // it may not be read from again.
  void Cancel() && { callback_ = nullptr; }

 private:
  std::function<void()> callback_;
};

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

void ModelMetadataPopulator::LoadMetadata(
    const tflite::ModelMetadataT& model_metadata_t) {
  // Pack the model metadata in a buffer.
  flatbuffers::FlatBufferBuilder model_metadata_fbb;
  model_metadata_fbb.Finish(
      tflite::ModelMetadata::Pack(model_metadata_fbb, &model_metadata_t),
      tflite::ModelMetadataIdentifier());
  auto model_metadata_buffer = std::make_unique<tflite::BufferT>();
  model_metadata_buffer->data = {
      model_metadata_fbb.GetBufferPointer(),
      model_metadata_fbb.GetBufferPointer() + model_metadata_fbb.GetSize()};
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
ModelMetadataPopulator::AppendAssociatedFiles(const char* buffer_data,
                                              size_t buffer_size) {
  // Initialize zip archive.
  mz_zip_archive zip_archive;
  memset(&zip_archive, 0, sizeof(zip_archive));
  // Open for writing, reserving `buffer_size` at beginning of file.
  if (!mz_zip_writer_init_heap(&zip_archive, buffer_size, 0)) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Unable to create zip archive",
        TfLiteSupportStatus::kMetadataAssociatedFileZipError);
  }
  // Automatically close the zip archive upon exiting this function, whether at
  // completion or due to an error.
  auto zip_writer_cleanup =
      SimpleCleanUp([&zip_archive] { mz_zip_writer_end(&zip_archive); });
  // Write associated files.
  for (const auto& pair : associated_files_) {
    if (!mz_zip_writer_add_mem(&zip_archive, pair.first.c_str(),
                               pair.second.data(), pair.second.length(),
                               MZ_DEFAULT_COMPRESSION)) {
      return CreateStatusWithPayload(
          StatusCode::kUnknown,
          absl::StrFormat("Unable to append associated file %s", pair.first),
          TfLiteSupportStatus::kMetadataAssociatedFileZipError);
    }
  }
  // Finalize. Note `mz_zip_writer_finalize_zip_archive()` handles allocating
  // `zip_buffer_data`, which thus needs to be free'd down the road.
  void* zip_buffer_data;
  size_t zip_buffer_size;
  if (!mz_zip_writer_finalize_heap_archive(&zip_archive, &zip_buffer_data,
                                           &zip_buffer_size)) {
    return CreateStatusWithPayload(
        StatusCode::kUnknown, "Unable to finalize zip archive",
        TfLiteSupportStatus::kMetadataAssociatedFileZipError);
  }
  // Copy model buffer in reserved space and build result.
  memcpy(zip_buffer_data, buffer_data, buffer_size);
  std::string final_buffer(reinterpret_cast<char*>(zip_buffer_data),
                           zip_buffer_size);
  // Cleanup and return.
  free(zip_buffer_data);
  return final_buffer;
}

tflite::support::StatusOr<std::string> ModelMetadataPopulator::Populate() {
  flatbuffers::FlatBufferBuilder model_fbb;
  model_fbb.Finish(tflite::Model::Pack(model_fbb, &model_t_),
                   tflite::ModelIdentifier());
  return AppendAssociatedFiles(
      reinterpret_cast<char*>(model_fbb.GetBufferPointer()),
      model_fbb.GetSize());
}

}  // namespace metadata
}  // namespace tflite
