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
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils_c.h"


// These need to be defined for stb_image.h and stb_image_write.h to include
// the actual implementations of image decoding/encoding functions.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"



struct ImageData DecodeImageFromFile(const char *file_name) {
  struct ImageData image_data;
  image_data.pixel_data = stbi_load(file_name, &image_data.width,
                                    &image_data.height, &image_data.channels,
                                    /*desired_channels=*/0);
  // if (image_data.pixel_data == NULL) {
  //     return NULL;
  // }

  if (image_data.channels != 1 && image_data.channels != 3 &&
      image_data.channels != 4) {
      // stbi_image_free(image_data.pixel_data);
      return image_data;
  }

  return image_data;

}

void ImageDataFree(struct ImageData* image) { stbi_image_free(image->pixel_data); }
