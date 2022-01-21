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
#import "tensorflow_lite_support/ios/test/task/vision/object_detector/utils/sources/TFLTestUtils.h"
#import "tensorflow_lite_support/ios/test/task/vision/utils/sources/GMLImage+Helpers.h"

@interface TFLObjectDetectorTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@end

@implementation TFLObjectDetectorTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class.
  [super setUp];
  self.modelPath = [[NSBundle bundleForClass:[self class]]
      pathForResource:@"coco_ssd_mobilenet_v1_1.0_quant_2018_06_29"
               ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);
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

  TFLDetectionResult *detectionResult = [objectDetector detectWithGMLImage:gmlImage error:nil];
  [TFLTestUtils verifyDetectionResult:detectionResult];
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

  TFLDetectionResult *detectionResult = [objectDetector detectWithGMLImage:gmlImage error:nil];

  XCTAssertLessThanOrEqual([detectionResult.detections count], maxResults);

  [TFLTestUtils verifyDetection:detectionResult.detections[0]
            expectedBoundingBox:CGRectMake(54, 396, 393, 199)
             expectedFirstScore:0.632812
             expectedFirstLabel:@"cat"];

  [TFLTestUtils verifyDetection:detectionResult.detections[1]
            expectedBoundingBox:CGRectMake(602, 157, 394, 447)
             expectedFirstScore:0.609375
             expectedFirstLabel:@"cat"];

  [TFLTestUtils verifyDetection:detectionResult.detections[2]
            expectedBoundingBox:CGRectMake(260, 394, 179, 209)
             expectedFirstScore:0.5625
             expectedFirstLabel:@"cat"];
}

@end
