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
#import "tensorflow_lite_support/ios/test/task/vision/object_detector/utils/sources/TFLTestUtils.h"
#import <XCTest/XCTest.h>

@implementation TFLTestUtils

+ (void)verifyDetectionResult:(TFLDetectionResult *)detectionResult {
  XCTAssertGreaterThan([detectionResult.detections count], 0);

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

  [TFLTestUtils verifyDetection:detectionResult.detections[3]
            expectedBoundingBox:CGRectMake(387, 197, 281, 409)
             expectedFirstScore:0.488281
             expectedFirstLabel:@"dog" ];
}

+ (void) verifyDetection:(TFLDetection *)detection
     expectedBoundingBox:(CGRect) expectedBoundingBox
      expectedFirstScore:(float) expectedFirstScore
      expectedFirstLabel:(NSString *)expectedFirstLabel {
  XCTAssertGreaterThan([detection.categories count], 0);
  XCTAssertEqual(detection.boundingBox.origin.x, expectedBoundingBox.origin.x);
  XCTAssertEqual(detection.boundingBox.origin.y, expectedBoundingBox.origin.y);
  XCTAssertEqual(detection.boundingBox.size.width, expectedBoundingBox.size.width);
  XCTAssertEqual(detection.boundingBox.size.height, expectedBoundingBox.size.height);
  XCTAssertEqualObjects(detection.categories[0].label, expectedFirstLabel);
  XCTAssertEqualWithAccuracy(detection.categories[0].score, expectedFirstScore, 0.001);
}

@end
