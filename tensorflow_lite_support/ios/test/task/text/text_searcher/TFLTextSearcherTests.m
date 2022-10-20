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

#import "tensorflow_lite_support/ios/task/text/sources/TFLTextSearcher.h"

NS_ASSUME_NONNULL_BEGIN

#define VerifySearchResultCount(searchResult, expectedNearestNeighborsCount) \
  XCTAssertEqual(searchResult.nearestNeighbors.count, expectedNearestNeighborsCount);

#define VerifyNearestNeighbor(nearestNeighbor, expectedMetadata, expectedDistance) \
  XCTAssertEqualObjects(nearestNeighbor.metadata, expectedMetadata);               \
  XCTAssertEqualWithAccuracy(nearestNeighbor.distance, expectedDistance, 1e-6);

@interface TFLTextSearcherTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@end

@implementation TFLTextSearcherTests

- (void)setUp {
  [super setUp];
  self.modelPath =
      [[NSBundle bundleForClass:self.class] pathForResource:@"regex_searcher"
                                                     ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);
}

- (TFLTextSearcher *)testSuccessfulCreationOfTextSearcherWithSearchContent:(NSString *)modelPath {
  TFLTextSearcherOptions *textSearcherOptions =
      [[TFLTextSearcherOptions alloc] initWithModelPath:self.modelPath];

  TFLTextSearcher *textSearcher = [TFLTextSearcher textSearcherWithOptions:textSearcherOptions
                                                                         error:nil];
  XCTAssertNotNil(textSearcher);

  return textSearcher;
}

- (void)verifySearchResultForInferenceWithSearchContent:(TFLSearchResult *)searchResult {
  VerifySearchResultCount(searchResult,
                          5  // expectedNearestNeighborsCount
  );

  VerifyNearestNeighbor(searchResult.nearestNeighbors[0],
                        @"burger",  // expectedMetadata
                        198.456329  // expectedDistance
  );
  VerifyNearestNeighbor(searchResult.nearestNeighbors[1],
                        @"car",     // expectedMetadata
                        226.022186  // expectedDistance
  );
  VerifyNearestNeighbor(searchResult.nearestNeighbors[2],
                        @"bird",    // expectedMetadata
                        227.297668  // expectedDistance
  );
  VerifyNearestNeighbor(searchResult.nearestNeighbors[3],
                        @"dog",     // expectedMetadata
                        229.133789  // expectedDistance
  );
  VerifyNearestNeighbor(searchResult.nearestNeighbors[4],
                        @"cat",     // expectedMetadata
                        229.718948  // expectedDistance
  );
}

- (void)testSuccessfullInferenceWithSearchContentOnText {
  TFLTextSearcher *textSearcher =
      [self testSuccessfulCreationOfTextSearcherWithSearchContent:self.modelPath];
  // GMLImage *gmlImage =
  //     [GMLImage imageFromBundleWithClass:self.class fileName:@"burger" ofType:@"jpg"];
  // XCTAssertNotNil(gmlImage);

  TFLSearchResult *searchResult = [textSearcher searchWithText:@"The weather was excellent." error:nil];
  [self verifySearchResultForInferenceWithSearchContent:searchResult];
}

@end

NS_ASSUME_NONNULL_END
