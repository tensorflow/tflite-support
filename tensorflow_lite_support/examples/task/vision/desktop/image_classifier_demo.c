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

// Example usage:
// bazel run -c opt \
//  tensorflow_lite_support/examples/task/vision/desktop:image_classifier_demo \
//  -- \
//  --model_path=/path/to/model.tflite \
//  --image_path=/path/to/image.jpg

#include <stdio.h>
#include <stdint.h>

// Thhis need to be defined for stb_image.h to include
// the actual implementations of image decoding/encoding functions.
#define STB_IMAGE_IMPLEMENTATION

#include "tensorflow_lite_support/cc/task/vision/image_classifier_c_api.h"
#include "stb_image.h"
#include "stb_image_write.h"


struct ImageClassifierOptions* BuildImageClassifierOptions(const char* model_path) {

  ImageClassifierOptions *options = ImageClassifierOptionsCreate();
  ImageClassifierOptionsAddClassNameBlackList(options, "/m/01bwbt");
  ImageClassifierOptionsAddClassNameBlackList(options, "/m/0bwm6m");
  ImageClassifierOptionsSetScoreThreshold(options, 0);
  ImageClassifierOptionsSetMaxResults(options, 5);
  ImageClassifierOptionsSetModelFilePath(options, model_path);

  return options;
}
 
void DisplayClassificationResults(struct ClassificationResult *classification_result) {

  for (int head = 0; head < classification_result->size; ++head) {
      struct Classifications test_classification = classification_result->classifications[head];
      for (int rank = 0; rank < test_classification.size; ++rank) {
          printf("  Rank #%d:\n", rank);
          printf("   index       : %d\n",
                                        test_classification.classes[rank].index);
          printf("   score       : %.5f\n",
                                        test_classification.classes[rank].score);

          printf("   display name: %s\n",
                                  test_classification.classes[rank].display_name);
          printf("   class name  : %s\n",
                                          test_classification.classes[rank].class_name);
    }
  }
}

void Classify(const char* model_path, const char* image_path) {
  struct ImageClassifierOptions *options = BuildImageClassifierOptions(model_path);
  ImageClassifier *image_classifier = ImageClassifierFromOptions(options);
  if (image_classifier == NULL) {
    printf("An error occured while instantiating the Image Classifier\n");
    return;
  }

  int image_width;
  int image_height;
  int channels;

  uint8_t *pixel_data = stbi_load(image_path, &image_width,
                                    &image_height, &channels,
                                    /*desired_channels=*/0);
  
  printf("%p\n",pixel_data);
  struct FrameBuffer frame_buffer = {.dimension.width = image_width, 
                                    .dimension.height = image_height, 
                                    .plane.buffer = pixel_data, 
                                    .plane.stride.row_stride_bytes = image_width  * channels, 
                                    .plane.stride.pixel_stride_bytes = channels, 
                                    .format = kRGB};


  struct ClassificationResult *classification_result = ImageClassifierClassify(image_classifier, &frame_buffer);
  
  if (classification_result == NULL) {
    printf("An error occured while classifying the image\n");
    return;
  }
  
  DisplayClassificationResults(classification_result);
  
  ImageClassifierDelete(image_classifier);
  ImageClassifierClassificationResultDelete(classification_result);

}

int main(int argc, char** argv) {
  const char *model_path = "/tmp/aiy_vision_classifier_birds_V1_3.tflite";
  const char *image_path = "/Users/priankakariat/Documents/Projects/TensorFlow/tflite-support/tensorflow_lite_support/examples/task/vision/desktop/g3doc/sparrow.jpg";
  Classify(model_path, image_path);
  return 0;

}