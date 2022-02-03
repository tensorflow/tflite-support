/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#import <XCTest/XCTest.h>

#import "third_party/tensorflow_lite_support/ios/task/vision/utils/sources/GMLImageUtils.h"
#import "third_party/tensorflow_lite_support/ios/test/task/vision/utils/sources/GMLImage+Helpers.h"

#include "third_party/tensorflow_lite_support/c/task/vision/utils/frame_buffer_cpp_c_utils.h"
#include "third_party/tensorflow_lite_support/cc/task/vision/image_classifier.h"

using ImageClassifier = ::tflite::task::vision::ImageClassifier;
using ImageClassifierOptions = ::tflite::task::vision::ImageClassifierOptions;
using ClassificationResult = ::tflite::task::vision::ClassificationResult;

@interface TFLImageClassifierCoreMLDelegateTest : XCTestCase {
  NSString* _modelPath;
}
@end

@implementation TFLImageClassifierCoreMLDelegateTest

- (void)setUp {
  [super setUp];

  // This image classifier can mostly be deoplyed through CoreML. Below is from the delegate logs:
  // "INFO: CoreML delegate: 64 nodes delegated out of 66 nodes, with 2 partitions."
  _modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"mobilenet_v2_1.0_224"
                                                                ofType:@"tflite"];
  XCTAssertNotNil(_modelPath);
}

- (void)testCoreMLDelegateCreationFailsWithNeuralEngine {
  // Configures the options.
  ImageClassifierOptions options;
  options.mutable_base_options()->mutable_model_file()->set_file_name(_modelPath.UTF8String);
  options.mutable_base_options()
      ->mutable_compute_settings()
      ->mutable_tflite_settings()
      ->set_delegate(::acceleration::Delegate::CORE_ML);
  options.mutable_base_options()
      ->mutable_compute_settings()
      ->mutable_tflite_settings()
      ->mutable_coreml_settings()
      ->set_enabled_devices(::acceleration::CoreMLDelegateSettings::DEVICES_ALL);

  // Creates the classifier.
  tflite::support::StatusOr<std::unique_ptr<ImageClassifier>> image_classifier_status =
      ImageClassifier::CreateFromOptions(options);
  XCTAssertTrue(image_classifier_status.ok());
  const std::unique_ptr<ImageClassifier>& image_classifier = image_classifier_status.value();
  XCTAssertNotEqual(image_classifier.get(), nullptr);

  // Loads the test image.
  GMLImage* gmlImage = [GMLImage imageFromBundleWithClass:[self class]
                                                 fileName:@"burger"
                                                   ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  // Converts the test image to a frame buffer.
  NSError* error;
  TfLiteFrameBuffer* cFrameBuffer = [GMLImageUtils cFrameBufferWithGMLImage:gmlImage error:&error];
  XCTAssertNotEqual(cFrameBuffer, nullptr);
  tflite::support::StatusOr<std::unique_ptr<::tflite::task::vision::FrameBuffer>>
      frame_buffer_status = ::tflite::task::vision::CreateCppFrameBuffer(cFrameBuffer);
  XCTAssertTrue(frame_buffer_status.ok());
  const ::tflite::task::vision::FrameBuffer& frame_buffer = *frame_buffer_status.value();

  // Classifies the image.
  tflite::support::StatusOr<ClassificationResult> classification_result_status =
      image_classifier->Classify(frame_buffer);
  XCTAssertTrue(classification_result_status.ok());
  const ClassificationResult& classification_result = classification_result_status.value();

  // Retrieves the top class.
  XCTAssertGreaterThan(classification_result.classifications_size(), 0);
  const ::tflite::task::vision::Classifications& classification =
      classification_result.classifications(0);
  XCTAssertGreaterThan(classification.classes_size(), 0);
  const ::tflite::task::vision::Class& topClass = classification.classes(0);

  // Verifies the class name & score.
  NSString* className = [NSString stringWithCString:topClass.class_name().c_str()];
  XCTAssertEqualObjects(className, @"cheeseburger");
  XCTAssertEqualWithAccuracy(topClass.score(), 0.748976, 0.001);
}

@end
