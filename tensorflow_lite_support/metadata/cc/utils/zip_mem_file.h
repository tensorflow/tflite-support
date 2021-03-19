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

#ifndef TENSORFLOW_LITE_SUPPORT_METADATA_CC_UTILS_ZIP_MEM_FILE_H_
#define TENSORFLOW_LITE_SUPPORT_METADATA_CC_UTILS_ZIP_MEM_FILE_H_

#include <cstdlib>

#include "absl/strings/string_view.h"
#include "contrib/minizip/ioapi.h"

namespace tflite {
namespace metadata {

// In-memory zip file implementation.
//
// Adapted from [1], with a few key differences:
// * backed by an `std::string` instead of malloc-ed C buffers,
// * supports opening the file for writing through `zipOpen2`.
//
// [1]:
// https://github.com/google/libkml/blob/master/third_party/zlib-1.2.3/contrib/minizip/iomem_simple.c
class ZipMemFile {
 public:
  // Constructs an in-memory zip file from a buffer.
  ZipMemFile(const char* buffer, size_t size);
  // Provides access to the `zlib_filefunc_def` implementation for the in-memory
  // zip file.
  zlib_filefunc_def& GetFileFuncDef();
  // Provides access to the file contents.
  absl::string_view GetFileContent() const;

 private:
  // The string backing the in-memory file.
  std::string data_;
  // The current offset in the file.
  size_t offset_;
  // The `zlib_filefunc_def` implementation for this in-memory zip file.
  zlib_filefunc_def zlib_filefunc_def_;

  // Convenience function to access the current data size.
  size_t Size() const { return data_.size(); }

  // The file function implementations used in the `zlib_filefunc_def`.
  static voidpf OpenFile(voidpf opaque, const char* filename, int mode);
  static size_t ReadFile(voidpf opaque, voidpf stream, void* buf, size_t size);
  static size_t WriteFile(voidpf opaque, voidpf stream, const void* buf,
                          size_t size);
  static ptrdiff_t TellFile(voidpf opaque, voidpf stream);
  static ptrdiff_t SeekFile(voidpf opaque, voidpf stream, size_t offset,
                            int origin);
  static int CloseFile(voidpf opaque, voidpf stream);
  static int ErrorFile(voidpf opaque, voidpf stream);
};

}  // namespace metadata
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_METADATA_CC_UTILS_ZIP_MEM_FILE_H_
