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

#import "tensorflow_lite_support/ios/task/vision/sources/TFLImageEmbedder.h"
#include "tensorflow_lite_support/cc/task/vision/image_embedder.h"

namespace {
using ImageEmbedderCpp = ::tflite::task::vision::ImageEmbedder;
using ImageEmbedderOptionsCpp =
    ::tflite::task::vision::ImageEmbedderOptions;
using ::tflite::support::StatusOr;


}  

@implementation TFLImageEmbedder {
    std::unique_ptr<ImageEmbedderCpp> cppImageEmbedder;
}

- (instancetype)initWithModelPath:(NSString *)modelPath {
  self = [super init];
  if (self) {
    ImageEmbedderOptionsCpp cc_options;

    cc_options.mutable_model_file_with_metadata()->set_file_name(modelPath.UTF8String);

    StatusOr<std::unique_ptr<ImageEmbedderCpp>> embedder_status =
      ImageEmbedderCpp::CreateFromOptions(cc_options);
 
    if (embedder_status.ok()) {
        cppImageEmbedder = std::move(embedder_status.value());
    }
    else {
       return nil;
    }
  }
  return self;
}

@end
