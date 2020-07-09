/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_SUPPORT_CC_UTILS_COMMON_UTILS_H_
#define TENSORFLOW_LITE_SUPPORT_CC_UTILS_COMMON_UTILS_H_

#include <string>
#include <vector>

namespace tflite {
namespace support {
namespace utils {

// read a vocab file, create a vector of strings
std::vector<std::string> LoadVocabFromFile(const std::string& path_to_vocab);

std::vector<std::string> LoadVocabFromBuffer(const char* vocab_buffer_data,
                                             const size_t vocab_buffer_size);
}  // namespace utils
}  // namespace support
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_UTILS_COMMON_UTILS_H_
