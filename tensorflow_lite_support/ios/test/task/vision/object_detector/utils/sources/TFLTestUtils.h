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
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

#define VerifyDetection(detection, expectedBoundingBox, expectedFirstScore, expectedFirstLabel) \
  XCTAssertGreaterThan([detection.categories count], 0);                                        \
  NSLog(@"Detected %f", detection.categories[0].score);                                         \
  NSLog(@"Expected %f", expectedFirstScore);                                                    \
  XCTAssertEqual(detection.boundingBox.origin.x, expectedBoundingBox.origin.x);                 \
  XCTAssertEqual(detection.boundingBox.origin.y, expectedBoundingBox.origin.y);                 \
  XCTAssertEqual(detection.boundingBox.size.width, expectedBoundingBox.size.width);             \
  XCTAssertEqual(detection.boundingBox.size.height, expectedBoundingBox.size.height);           \
  XCTAssertEqualObjects(detection.categories[0].label, expectedFirstLabel);                     \
  XCTAssertEqualWithAccuracy(detection.categories[0].score, expectedFirstScore, 0.001)


NS_ASSUME_NONNULL_END
