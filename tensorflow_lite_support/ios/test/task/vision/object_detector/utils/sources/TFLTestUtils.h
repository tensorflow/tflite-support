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
#import "tensorflow_lite_support/ios/task/processor/sources/TFLDetectionResult.h"
NS_ASSUME_NONNULL_BEGIN

@interface TFLTestUtils : NSObject

+ (void)verifyDetectionResult:(TFLDetectionResult *)detectionResult
    NS_SWIFT_NAME(verify(detectionResult:));

+ (void)verifyDetection:(TFLDetection *)detection
    expectedBoundingBox:(CGRect)expectedBoundingBox
     expectedFirstScore:(float)expectedFirstScore
     expectedFirstLabel:(NSString *)expectedFirstLabel
    NS_SWIFT_NAME(verify(detection:expectedBoundingBox:expectedFirstScore:expectedFirstLabel:));

@end

NS_ASSUME_NONNULL_END
