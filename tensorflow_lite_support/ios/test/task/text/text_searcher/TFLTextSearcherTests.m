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

#define ValidateSearchResultCount(searchResult, expectedNearestNeighborsCount) \
  XCTAssertEqual(searchResult.nearestNeighbors.count, expectedNearestNeighborsCount);

#define ValidateNearestNeighbor(nearestNeighbor, expectedMetadata, expectedDistance) \
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

- (TFLTextSearcher *)textSearcherWithSearcherModelPath:(NSString *)modelPath {
  TFLTextSearcherOptions *textSearcherOptions =
      [[TFLTextSearcherOptions alloc] initWithModelPath:self.modelPath];

  TFLTextSearcher *textSearcher = [TFLTextSearcher textSearcherWithOptions:textSearcherOptions
                                                                         error:nil];
  XCTAssertNotNil(textSearcher);

  return textSearcher;
}

- (void)validateSearchResultForInferenceWithSearchContent:(TFLSearchResult *)searchResult {
  ValidateSearchResultCount(searchResult,
                          5  // expectedNearestNeighborsCount
  );

  ValidateNearestNeighbor(searchResult.nearestNeighbors[0],
                          @"The weather was excellent.",  // expectedMetadata
                          0.889664649963  // expectedDistance
  );
  ValidateNearestNeighbor(searchResult.nearestNeighbors[1],
                          @"The sun was shining on that day.",     // expectedMetadata
                          0.889667928219  // expectedDistance
  );
  ValidateNearestNeighbor(searchResult.nearestNeighbors[2],
                          @"The cat is chasing after the mouse.",    // expectedMetadata
                          0.889669716358  // expectedDistance
  );
  ValidateNearestNeighbor(searchResult.nearestNeighbors[3],
                          @"It was a sunny day.",     // expectedMetadata
                          0.889671087265 // expectedDistance
  );
  ValidateNearestNeighbor(searchResult.nearestNeighbors[4],
                          @"He was very happy with his newly bought car.",     // expectedMetadata
                          0.889671683311  // expectedDistance
  );
}

- (void)testSuccessfullInferenceWithSearchContentOnText {
  TFLTextSearcher *textSearcher =
      [self textSearcherWithSearcherModelPath:self.modelPath];
 
  TFLSearchResult *searchResult = [textSearcher searchWithText:@"The weather was excellent." error:nil];
  [self validateSearchResultForInferenceWithSearchContent:searchResult];
}

@end

NS_ASSUME_NONNULL_END
