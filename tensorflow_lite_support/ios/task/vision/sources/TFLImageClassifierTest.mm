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

#import "tensorflow_lite_support/ios/task/vision/sources/TFLImageClassifierTest.h"
#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"

namespace {
using ImageClassifierCpp = ::tflite::task::vision::ImageClassifier;
using ImageClassifierOptionsCpp =
    ::tflite::task::vision::ImageClassifierOptions;
using ::tflite::support::StatusOr;


}  // namespace

@interface TFLImageClassifierTest ()
/** ImageClassifier backed by C API */
// @property(nonatomic) TfLiteImageClassifier *imageClassifier;
@end

@implementation TFLImageClassifierTest

-(BOOL)testClassifier {

ImageClassifierOptionsCpp cc_options;

  cc_options.mutable_base_options()->mutable_model_file()->set_file_name(
      "");

  StatusOr<std::unique_ptr<ImageClassifierCpp>> classifier_status =
      ImageClassifierCpp::CreateFromOptions(cc_options);

  return classifier_status.ok();

}

@end
