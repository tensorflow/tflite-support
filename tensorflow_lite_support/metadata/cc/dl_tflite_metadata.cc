#include "tensorflow_lite_support/metadata/cc/dl_tflite_metadata.h"

#include <memory>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "tensorflow_lite_support/metadata/cc/metadata_extractor.h"

namespace tflite {
namespace metadata {


const char* get_version(void* buffer_data, size_t buffer_size) {
  const char* char_buffer = reinterpret_cast<const char *>(buffer_data);
  support::StatusOr<std::unique_ptr<ModelMetadataExtractor>> status_meta_ptr =
    ModelMetadataExtractor::CreateFromModelBuffer(char_buffer, buffer_size);
  if (status_meta_ptr.status().ok()) {
      const tflite::ModelMetadata* meta_data =
        (*status_meta_ptr)->GetModelMetadata();
      if (meta_data and meta_data->version()) {
        return meta_data->version()->c_str();
      }
  }
  return "";
}

}  // namespace metadata
}  // namespace tflite
