#include "tensorflow_lite_support/metadata/cc/dl_tflite_metadata.h"

#include <iostream>
#include <memory>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "tensorflow_lite_support/metadata/cc/metadata_extractor.h"

namespace tflite {
namespace metadata {


std::string get_version(const char* buffer_data, size_t buffer_size) {

  support::StatusOr<std::unique_ptr<ModelMetadataExtractor>> status_meta_ptr =
    ModelMetadataExtractor::CreateFromModelBuffer(buffer_data, buffer_size);
  if (status_meta_ptr.status().ok()) {
      const tflite::ModelMetadata* meta_data =
        (*status_meta_ptr)->GetModelMetadata();
      if (meta_data and meta_data->version()) {
        return meta_data->version()->str();
      }
  }
  return std::string();
}

}  // namespace metadata
}  // namespace tflite
