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
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImage+Utils.h"

#define VerifyCategory(category, expectedIndex, expectedScore, expectedlLabel, \
                       expectedDisplayName)                                    \
  XCTAssertEqual(category.classIndex, expectedIndex);                          \
  XCTAssertEqualWithAccuracy(category.score, expectedScore, 1e-6);             \
  XCTAssertEqualObjects(category.label, expectedlLabel);                       \
  XCTAssertEqualObjects(category.displayName, expectedDisplayName);

#define VerifyClassifications(classifications, expectedHeadIndex, expectedCategoryCount) \
  XCTAssertEqual(classifications.categories.count, expectedCategoryCount);               \
  XCTAssertEqual(classifications.headIndex, expectedHeadIndex)

#define VerifyClassificationResult(classificationResult, expectedClassificationsCount) \
  XCTAssertEqual(classificationResult.classifications.count, expectedClassificationsCount)

NS_ASSUME_NONNULL_BEGIN

@interface TFLImageClassifierTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@end

@implementation TFLImageClassifierTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class.
  [super setUp];
  self.modelPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"mobilenet_v2_1.0_224"
                                                                    ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);
}

- (void)testSuccessfullImageInferenceOnMLImageWithUIImage {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"burger" ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResult = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                  error:nil];
  const NSInteger categoryCount = 1001;
  VerifyClassificationResult(classificationResult, 1);
  VerifyClassifications(classificationResult.classifications[0], 0, categoryCount);
  VerifyCategory(classificationResult.classifications[0].categories[0], 934, 0.748976,
                 @"cheeseburger", nil);
  VerifyCategory(classificationResult.classifications[0].categories[1], 925, 0.024646, @"guacamole",
                 nil);
  VerifyCategory(classificationResult.classifications[0].categories[2], 932, 0.022505, @"bagel",
                 nil);
}

- (void)testModelOptionsWithMaxResults {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];
  int maxResults = 3;
  imageClassifierOptions.classificationOptions.maxResults = maxResults;

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"burger" ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResult = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                  error:nil];

  VerifyClassificationResult(classificationResult, 1);
  VerifyClassifications(classificationResult.classifications[0], 0, maxResults);
  VerifyCategory(classificationResult.classifications[0].categories[0], 934, 0.748976,
                 @"cheeseburger", nil);
  VerifyCategory(classificationResult.classifications[0].categories[1], 925, 0.024646, @"guacamole",
                 nil);
  VerifyCategory(classificationResult.classifications[0].categories[2], 932, 0.022505, @"bagel",
                 nil);
}

- (void)testInferenceWithBoundingBox {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];
  int maxResults = 3;
  imageClassifierOptions.classificationOptions.maxResults = maxResults;

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"multi_objects" ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  CGRect roi = CGRectMake(406, 110, 148, 153);
  TFLClassificationResult *classificationResult = [imageClassifier classifyWithGMLImage:gmlImage
                                                                       regionOfInterest:roi
                                                                                  error:nil];

  VerifyClassificationResult(classificationResult, 1);
  VerifyClassifications(classificationResult.classifications[0], 0, maxResults);
  // TODO: match the label and score as image_classifier_test.cc
  VerifyCategory(classificationResult.classifications[0].categories[0], 806, 0.997143,
                 @"soccer ball", nil);
  VerifyCategory(classificationResult.classifications[0].categories[1], 891, 0.000380,
                 @"volleyball", nil);
  VerifyCategory(classificationResult.classifications[0].categories[2], 685, 0.000198, @"ocarina",
                 nil);
}

- (void)testInferenceWithRGBAImage {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"burger" ofType:@"png"];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResult = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                  error:nil];

  VerifyCategory(classificationResult.classifications[0].categories[0], 934, 0.738065,
                 @"cheeseburger", nil);
  VerifyCategory(classificationResult.classifications[0].categories[1], 925, 0.027371, @"guacamole",
                 nil);
  VerifyCategory(classificationResult.classifications[0].categories[2], 932, 0.026174, @"bagel",
                 nil);
}

@end

NS_ASSUME_NONNULL_END
