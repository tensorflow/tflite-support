#ifndef TENSORFLOW_LITE_SUPPORT_METADATA_CC_DL_TFLITE_METADATA_H_
#define TENSORFLOW_LITE_SUPPORT_METADATA_CC_DL_TFLITE_METADATA_H_

#include <string>

namespace tflite {
namespace metadata {


// Retrieves a tflite model's version information from the
// ModelMetadata. If not set, an empty string is returned.
// In future, we may want to move this to a class-based implementation
// if other metadata need to be exposed.
//
// Args:
//    buffer_data: The tflite model's memory-mapped address
//    buffer_size: The size of the buffer.

std::string get_version(const char* buffer_data, size_t buffer_size);

}  // namespace metadata
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_METADATA_CC_DL_TFLITE_METADATA_H_
