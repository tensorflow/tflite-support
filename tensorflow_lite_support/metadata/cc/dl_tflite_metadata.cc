#include "tensorflow_lite_support/metadata/cc/dl_tflite_metadata.h"

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow_lite_support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace metadata {

namespace {

  constexpr char kMetadataBufferName[] = "TFLITE_METADATA";

}  // namespace

const char* get_version(void* buffer_data, size_t buffer_size) {
  flatbuffers::Verifier verifier = flatbuffers::Verifier(
      reinterpret_cast<const uint8_t *>(buffer_data), buffer_size);
  if (!tflite::VerifyModelBuffer(verifier)) {
    return "";
  }
  const tflite::Model* model = tflite::GetModel(buffer_data);
  if (model->metadata() == nullptr) {
    return "";
  }
  // Look for the "TFLITE_METADATA" field, if any.
  for (int i = 0; i < model->metadata()->size(); ++i) {
    const auto metadata = model->metadata()->Get(i);
    if (metadata->name()->str() != kMetadataBufferName) {
      continue;
    }
    const auto buffer_index = metadata->buffer();
    const auto metadata_buffer =
        model->buffers()->Get(buffer_index)->data()->data();
    if (!tflite::ModelMetadataBufferHasIdentifier(metadata_buffer)) {
      return "";
    }
    const tflite::ModelMetadata* meta_data = tflite::GetModelMetadata(metadata_buffer);
    if (meta_data and meta_data->version()) {
      return meta_data->version()->c_str();
    }
  }
  return "";
}

}  // namespace metadata
}  // namespace tflite
