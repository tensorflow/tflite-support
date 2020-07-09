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

#include "tensorflow_lite_support/cc/utils/common_utils.h"

#include <fstream>

namespace tflite {
namespace support {
namespace utils {

struct membuf : std::streambuf {
  membuf(char* begin, char* end) { this->setg(begin, begin, end); }
};

std::vector<std::string> LoadVocabFromFile(const std::string& path_to_vocab) {
  std::vector<std::string> vocab_from_file;
  std::ifstream in(path_to_vocab.c_str());
  std::string str;
  while (std::getline(in, str)) {
    if (!str.empty()) vocab_from_file.push_back(str);
  }
  in.close();

  return vocab_from_file;
}

std::vector<std::string> LoadVocabFromBuffer(const char* vocab_buffer_data,
                                             const size_t vocab_buffer_size) {
  membuf sbuf(const_cast<char*>(vocab_buffer_data),
              const_cast<char*>(vocab_buffer_data + vocab_buffer_size));
  std::vector<std::string> vocab_from_file;
  std::istream in(&sbuf);
  std::string str;
  while (std::getline(in, str)) {
    if (!str.empty()) vocab_from_file.push_back(str);
  }
  return vocab_from_file;
}

}  // namespace utils
}  // namespace support
}  // namespace tflite
