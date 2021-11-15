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
#import <XCTest/XCTest.h>
#import "tensorflow_lite_support/ios/task/vision/sources/TFLImageClassifier.h"

NS_ASSUME_NONNULL_BEGIN

@interface TFLImageClassifierTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@end

@implementation TFLImageClassifierTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class. static let bundle = Bundle(for: TFLSentencepieceTokenizerTest.self)
  self.modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"mobilenet_v2_1.0_224"
                                                                    ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);
}

- (void)tearDown {
  // Put teardown code here. This method is called after the invocation of each test method in the
  // class.
}

- (void)testSuccessfullImageInferenceOnMLImageWithUIImage {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];
  
  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  NSString *imagePath = [[NSBundle bundleForClass:[self class]] pathForResource:@"burger_crop"
                                                                         ofType:@"jpg"];
  UIImage *image = [[UIImage alloc] initWithContentsOfFile:imagePath];
  XCTAssertNotNil(image);

  GMLImage *gmlImage = [[GMLImage alloc] initWithIma,sge:image];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResults = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                   error:nil];
  XCTAssertTrue([classificationResults.classifications count] > 0);
  XCTAssertTrue([classificationResults.classifications[0].categories count] > 0);

  TFLCategory *category = classificationResults.classifications[0].categories[0];
  XCTAssertTrue([category.label isEqual:@"cheeseburger"]);
}

- (void)testModelOptionsWithMaxResults {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];
  int maxResults = 3;
  imageClassifierOptions.classificationOptions.maxResults = maxResults;

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  NSString *imagePath = [[NSBundle bundleForClass:[self class]] pathForResource:@"burger_crop"
                                                                         ofType:@"jpg"];
  UIImage *image = [[UIImage alloc] initWithContentsOfFile:imagePath];
  XCTAssertNotNil(image);

  GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResults = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                   error:nil];
  XCTAssertTrue([classificationResults.classifications count] > 0);
  XCTAssertTrue([classificationResults.classifications[0].categories count] > 0);

  TFLCategory *category = classificationResults.classifications[0].categories[0];
  XCTAssertTrue([category.label isEqual:@"cheeseburger"]);
}

- (void)testInferenceWithBoundingBox {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];
  int maxResults = 3;
  imageClassifierOptions.classificationOptions.maxResults = maxResults;

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  NSString *imagePath = [[NSBundle bundleForClass:[self class]] pathForResource:@"burger_crop"
                                                                         ofType:@"jpg"];
  UIImage *image = [[UIImage alloc] initWithContentsOfFile:imagePath];
  XCTAssertNotNil(image);

  GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];
  XCTAssertNotNil(gmlImage);

  CGRect roi = CGRectMake(20, 20, 200, 200);
  TFLClassificationResult *classificationResults = [imageClassifier classifyWithGMLImage:gmlImage
                                                                        regionOfInterest:roi
                                                                                   error:nil];
  XCTAssertTrue([classificationResults.classifications count] > 0);
  XCTAssertTrue([classificationResults.classifications[0].categories count] > 0);

  TFLCategory *category = classificationResults.classifications[0].categories[0];
  XCTAssertTrue([category.label isEqual:@"cheeseburger"]);
}

- (void)testInferenceWithRGBAImage {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];
  
  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  NSString *imagePath = [[NSBundle bundleForClass:[self class]] pathForResource:@"sparrow"
                                                                         ofType:@"png"];
  UIImage *image = [[UIImage alloc] initWithContentsOfFile:imagePath];
  XCTAssertNotNil(image);

  GMLImage *gmlImage = [[GMLImage alloc] initWithImage:image];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResults = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                   error:nil];
  XCTAssertTrue([classificationResults.classifications count] > 0);
  XCTAssertTrue([classificationResults.classifications[0].categories count] > 0);

  TFLCategory *category = classificationResults.classifications[0].categories[0];
  XCTAssertTrue([category.label isEqual:@"junco"]);
}

@end

NS_ASSUME_NONNULL_END
