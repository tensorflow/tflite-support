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

#define VerifyError(error, expectedDomain, expectedCode, expectedLocalizedDescription)  \
  XCTAssertNotNil(error);                                                               \
  XCTAssertEqual(error.domain, expectedDomain);                                         \
  XCTAssertEqual(error.code, expectedCode);                                             \
  XCTAssertNotEqual(                                                                    \
      [error.localizedDescription rangeOfString:expectedLocalizedDescription].location, \
      NSNotFound)

#define VerifyCategory(category, expectedIndex, expectedScore, expectedLabel, expectedDisplayName) \
  XCTAssertEqual(category.index, expectedIndex);                                                   \
  XCTAssertEqualWithAccuracy(category.score, expectedScore, 1e-6);                                 \
  XCTAssertEqualObjects(category.label, expectedLabel);                                            \
  XCTAssertEqualObjects(category.displayName, expectedDisplayName);

#define VerifyClassifications(classifications, expectedHeadIndex, expectedCategoryCount) \
  XCTAssertEqual(classifications.categories.count, expectedCategoryCount);               \
  XCTAssertEqual(classifications.headIndex, expectedHeadIndex)

#define VerifyClassificationResult(classificationResult, expectedClassificationsCount) \
  XCTAssertNotNil(classificationResult);                                               \
  XCTAssertEqual(classificationResult.classifications.count, expectedClassificationsCount)

static NSString *const expectedErrorDomain = @"org.tensorflow.lite.tasks";

NS_ASSUME_NONNULL_BEGIN

@interface TFLImageClassifierTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@end

@implementation TFLImageClassifierTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class.
  [super setUp];

  // Setting this property causes the tests to break after a test case fails.
  self.continueAfterFailure = NO;
  self.modelPath = [[NSBundle bundleForClass:self.class] pathForResource:@"mobilenet_v2_1.0_224"
                                                                    ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);
}

- (void)testInferenceOnMLImageWithUIImage {
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

  const NSInteger expectedClassificationsCount = 1;
  VerifyClassificationResult(classificationResult, expectedClassificationsCount);

  const NSInteger expectedHeadIndex = 0;
  const NSInteger expectedCategoryCount = 1001;
  VerifyClassifications(classificationResult.classifications[0], expectedHeadIndex,
                        expectedCategoryCount);
  VerifyCategory(classificationResult.classifications[0].categories[0],
                 934,              // expectedIndex
                 0.748976,         // expectedScore
                 @"cheeseburger",  // expectedLabel
                 nil               // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[1],
                 925,           // expectedIndex
                 0.024646,      // expectedScore
                 @"guacamole",  // expectedLabel
                 nil            // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[2],
                 932,       // expectedIndex
                 0.022505,  // expectedScore
                 @"bagel",  // expectedLabel
                 nil        // expectedDisplaName
  );
}

- (void)testErrorForSimultaneousClassNameBlackListAndWhiteList {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];

  imageClassifierOptions.classificationOptions.labelDenyList =
      [NSArray arrayWithObjects:@"cheeseburger", nil];
  imageClassifierOptions.classificationOptions.labelAllowList =
      [NSArray arrayWithObjects:@"bagel", nil];

  NSError *error = nil;
  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:&error];
  XCTAssertNil(imageClassifier);
  XCTAssertNotNil(error);

  const NSInteger expectedErrorCode = 2;
  NSString *const expectedLocalizedDescription =
      @"INVALID_ARGUMENT: `class_name_whitelist` and `class_name_blacklist` are mutually exclusive "
      @"options";
  VerifyError(error,
              expectedErrorDomain,          // expectedDomain
              expectedErrorCode,            // expectedCode
              expectedLocalizedDescription  // expectedLocalizedDescription
  );
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

  TFLClassificationResult *classificationResult = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                  error:nil];

  const NSInteger expectedClassificationsCount = 1;
  VerifyClassificationResult(classificationResult,
                             expectedClassificationsCount  // expectedClassificationsCount
  );

  const NSInteger expectedHeadIndex = 0;
  VerifyClassifications(classificationResult.classifications[0],
                        expectedHeadIndex,  // expectedHeadIndex
                        maxResults          // expectedCategoryCount
  );

  VerifyCategory(classificationResult.classifications[0].categories[0],
                 934,              // expectedIndex
                 0.748976,         // expectedScore
                 @"cheeseburger",  // expectedLabel
                 nil               // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[1],
                 925,           // expectedIndex
                 0.024646,      // expectedScore
                 @"guacamole",  // expectedLabel
                 nil            // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[2],
                 932,       // expectedIndex
                 0.022505,  // expectedScore
                 @"bagel",  // expectedLabel
                 nil        // expectedDisplaName
  );
}

- (void)testErrorForInvalidMaxResults {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];

  imageClassifierOptions.classificationOptions.maxResults = 0;

  NSError *error = nil;
  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:&error];
  XCTAssertNil(imageClassifier);
  XCTAssertNotNil(error);

  const NSInteger expectedErrorCode = 2;
  NSString *const expectedLocalizedDescription =
      @"INVALID_ARGUMENT: Invalid `max_results` option: value must be != 0";
  VerifyError(error,
              expectedErrorDomain,          // expectedDomain
              expectedErrorCode,            // expectedCode
              expectedLocalizedDescription  // expectedLocalizedDescription
  );
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

  const NSInteger expectedClassificationsCount = 1;
  VerifyClassificationResult(classificationResult,
                             expectedClassificationsCount  // expectedClassificationsCount
  );

  const NSInteger expectedHeadIndex = 0;
  VerifyClassifications(classificationResult.classifications[0],
                        expectedHeadIndex,  // expectedHeadIndex
                        maxResults          // expectedCategoryCount
  );

  // TODO: match the label and score as image_classifier_test.cc
  VerifyCategory(classificationResult.classifications[0].categories[0],
                 806,             // expectedIndex
                 0.997143,        // expectedScore
                 @"soccer ball",  // expectedLabel
                 nil              // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[1],
                 891,            // expectedIndex
                 0.000380,       // expectedScore
                 @"volleyball",  // expectedLabel
                 nil             // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[2],
                 685,         // expectedIndex
                 0.000198,    // expectedScore
                 @"ocarina",  // expectedLabel
                 nil          // expectedDisplaName
  );
}

- (void)testInferenceWithRGBAImage {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];

  int maxResults = 3;
  imageClassifierOptions.classificationOptions.maxResults = maxResults;

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"burger" ofType:@"png"];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResult = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                  error:nil];

  const NSInteger expectedClassificationsCount = 1;
  VerifyClassificationResult(classificationResult,
                             expectedClassificationsCount  // expectedClassificationsCount
  );

  const NSInteger expectedHeadIndex = 0;
  VerifyClassifications(classificationResult.classifications[0],
                        expectedHeadIndex,  // expectedHeadIndex
                        maxResults          // expectedCategoryCount
  );

  VerifyCategory(classificationResult.classifications[0].categories[0],
                 934,              // expectedIndex
                 0.738065,         // expectedScore
                 @"cheeseburger",  // expectedLabel
                 nil               // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[1],
                 925,           // expectedIndex
                 0.027371,      // expectedScore
                 @"guacamole",  // expectedLabel
                 nil            // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[2],
                 932,       // expectedIndex
                 0.026174,  // expectedScoree
                 @"bagel",  // expectedLabel
                 nil        // expectedDisplaName
  );
}

- (void)testErrorForNullGMLImage {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  NSError *error = nil;
  TFLClassificationResult *classificationResult = [imageClassifier classifyWithGMLImage:nil
                                                                                  error:&error];
  XCTAssertNil(classificationResult);

  const NSInteger expectedErrorCode = 2;
  NSString *const expectedLocalizedDescription =
      @"GMLImage argument cannot be nil.";
  VerifyError(error,
              expectedErrorDomain,          // expectedDomain
              expectedErrorCode,            // expectedCode
              expectedLocalizedDescription  // expectedLocalizedDescription
  );
}

- (void)testErrorForUnsupportedGMLImageSourceType {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  NSError *error = nil;

  // Initialize an invalid GMLImage with no source.
  GMLImage *gmlImage = [GMLImage new];
  TFLClassificationResult *classificationResult = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                  error:&error];
  XCTAssertNil(classificationResult);

  const NSInteger expectedErrorCode = 2;
  NSString *const expectedLocalizedDescription =
      @"Invalid source type for GMLImage.";
  VerifyError(error,
              expectedErrorDomain,          // expectedDomain
              expectedErrorCode,            // expectedCode
              expectedLocalizedDescription  // expectedLocalizedDescription
  );
}

@end

NS_ASSUME_NONNULL_END
