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

#import "tensorflow_lite_support/ios/task/vision/sources/TFLImageEmbedder.h"

NS_ASSUME_NONNULL_BEGIN

@interface TFLImageEmbedderTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@end

@implementation TFLImageEmbedderTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class.
  [super setUp];
  self.modelPath = [[NSBundle bundleForClass:self.class] pathForResource:@"mobilenet_v3_small_100_224_embedder"
                                                                  ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);
}

- (void)testInitSucceeds {
    TFLImageEmbedder *embedder = [[TFLImageEmbedder alloc] initWithModelPath:self.modelPath];
    XCTAssertNotNil(embedder);
}
@end

NS_ASSUME_NONNULL_END
