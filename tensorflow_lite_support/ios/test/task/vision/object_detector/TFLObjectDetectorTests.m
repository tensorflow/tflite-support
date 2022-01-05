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
#import <CoreGraphics/CoreGraphics.h>
#import <XCTest/XCTest.h>

#import "tensorflow_lite_support/ios/task/vision/sources/TFLObjectDetector.h"
#import "tensorflow_lite_support/ios/test/task/vision/utils/sources/GMLImage+Helpers.h"

@interface TFLObjectDetectorTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@end

@implementation TFLObjectDetectorTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class.
  self.modelPath = [[NSBundle bundleForClass:[self class]]
      pathForResource:@"coco_ssd_mobilenet_v1_1.0_quant_2018_06_29"
               ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);
}

- (void)tearDown {
  // Put teardown code here. This method is called after the invocation of each test method in the
  // class.
}

- (void)verfiyDetection:(TFLDetection *)detection
    expectedBoundingBox:(CGRect)expectedBoundingBox
     expectedFirstScore:(CGFloat)expectedFirstScore
     expectedFirstLabel:(NSString *)expectedFirstLabel {
  XCTAssertTrue([detection.categories count] > 0);
  NSLog(@"Detected %f", detection.categories[0].score);
  NSLog(@"Expected %f", expectedFirstScore);
  XCTAssertEqual(detection.boundingBox.origin.x, expectedBoundingBox.origin.x);
  XCTAssertEqual(detection.boundingBox.origin.y, expectedBoundingBox.origin.y);
  XCTAssertEqual(detection.boundingBox.size.width, expectedBoundingBox.size.width);
  XCTAssertEqual(detection.boundingBox.size.height, expectedBoundingBox.size.height);

  XCTAssertTrue([detection.categories[0].label isEqual:expectedFirstLabel]);
  XCTAssertTrue(detection.categories[0].score >= expectedFirstScore);
}

- (void)verifyResults:(TFLDetectionResult *)detectionResult {
  XCTAssertTrue([detectionResult.detections count] > 0);
  [self verfiyDetection:detectionResult.detections[0]
      expectedBoundingBox:CGRectMake(54, 396, 393, 199)
       expectedFirstScore:0.632812
       expectedFirstLabel:@"cat"];
  [self verfiyDetection:detectionResult.detections[1]
      expectedBoundingBox:CGRectMake(602, 157, 394, 447)
       expectedFirstScore:0.609375
       expectedFirstLabel:@"cat"];
  [self verfiyDetection:detectionResult.detections[2]
      expectedBoundingBox:CGRectMake(260, 394, 179, 209)
       expectedFirstScore:0.5625
       expectedFirstLabel:@"cat"];
  [self verfiyDetection:detectionResult.detections[3]
      expectedBoundingBox:CGRectMake(387, 197, 281, 409)
       expectedFirstScore:0.488281
       expectedFirstLabel:@"dog"];
}

- (void)verifyResultsWithMaxResultsOption:(TFLDetectionResult *)detectionResult
                               maxResults:(NSInteger)maxResults {
  XCTAssertTrue([detectionResult.detections count] <= maxResults);
  [self verfiyDetection:detectionResult.detections[0]
      expectedBoundingBox:CGRectMake(54, 396, 393, 199)
       expectedFirstScore:0.632812
       expectedFirstLabel:@"cat"];
  [self verfiyDetection:detectionResult.detections[1]
      expectedBoundingBox:CGRectMake(602, 157, 394, 447)
       expectedFirstScore:0.609375
       expectedFirstLabel:@"cat"];
  [self verfiyDetection:detectionResult.detections[2]
      expectedBoundingBox:CGRectMake(260, 394, 179, 209)
       expectedFirstScore:0.5625
       expectedFirstLabel:@"cat"];
}

- (void)testSuccessfullObjectDetectionOnMLImageWithUIImage {
  TFLObjectDetectorOptions *objectDetectorOptions =
      [[TFLObjectDetectorOptions alloc] initWithModelPath:self.modelPath];

  TFLObjectDetector *objectDetector =
      [TFLObjectDetector objectDetectorWithOptions:objectDetectorOptions error:nil];
  XCTAssertNotNil(objectDetector);

  GMLImage *gmlImage = [GMLImage imageFromBundleWithClass:[self class]
                                                 fileName:@"cats_and_dogs"
                                                   ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  TFLDetectionResult *detectionResults = [objectDetector detectWithGMLImage:gmlImage error:nil];
  [self verifyResults:detectionResults];
}

- (void)testModelOptionsWithMaxResults {
  TFLObjectDetectorOptions *objectDetectorOptions =
      [[TFLObjectDetectorOptions alloc] initWithModelPath:self.modelPath];
  int maxResults = 3;
  objectDetectorOptions.classificationOptions.maxResults = maxResults;

  TFLObjectDetector *objectDetector =
      [TFLObjectDetector objectDetectorWithOptions:objectDetectorOptions error:nil];
  XCTAssertNotNil(objectDetector);

  GMLImage *gmlImage = [GMLImage imageFromBundleWithClass:[self class]
                                                 fileName:@"cats_and_dogs"
                                                   ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  TFLDetectionResult *detectionResults = [objectDetector detectWithGMLImage:gmlImage error:nil];
  [self verifyResultsWithMaxResultsOption:detectionResults maxResults:maxResults];
}

@end
