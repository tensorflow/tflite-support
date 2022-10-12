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

#import "tensorflow_lite_support/ios/task/vision/sources/TFLImageSearcher.h"
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImage+Utils.h"

NS_ASSUME_NONNULL_BEGIN

NSString * const kSearcherModelName = @"mobilenet_v3_small_100_224_searcher";
NSString * const kEmbedderModelName = @"mobilenet_v3_small_100_224_searcher";
// NSString * const kMobileNetIndexName = @"searcher_index";
NSString * const kMobileNetIndexName = @"kk";


#define VerifySearchResultCount(searchResult, expectedNearestNeighborsCount) \
  XCTAssertEqual(searchResult.nearestNeighbors.count, expectedNearestNeighborsCount);

#define VerifyNearestNeighbor(nearestNeighbor, expectedMetadata, expectedDistance) \
  XCTAssertEqualObjects(nearestNeighbor.metadata, expectedMetadata);               \
  XCTAssertEqualWithAccuracy(nearestNeighbor.distance, expectedDistance, 1e-6);

@interface TFLImageSearcherTests : XCTestCase
@property(nonatomic, nullable) NSString *searcherModelPath;
@property(nonatomic, nullable) NSString *embedderModelPath;
@property(nonatomic, nullable) NSString *mobileNetIndexPath;
@end

@implementation TFLImageSearcherTests

- (void)setUp {
  [super setUp];
  self.searcherModelPath =
      [[NSBundle bundleForClass:self.class] pathForResource:kSearcherModelName
                                                     ofType:@"tflite"];
  XCTAssertNotNil(self.searcherModelPath);

   self.embedderModelPath =
      [[NSBundle bundleForClass:self.class] pathForResource:kEmbedderModelName
                                                     ofType:@"tflite"];
  XCTAssertNotNil(self.embedderModelPath);

  self.mobileNetIndexPath =
      [[NSBundle bundleForClass:self.class] pathForResource:kMobileNetIndexName
                                                     ofType:@"ldb"];
  XCTAssertNotNil(self.mobileNetIndexPath);
}

- (TFLImageSearcher *)testSuccessfulCreationOfImageSearcherWithSearchContent:(NSString *)modelPath {
  TFLImageSearcherOptions *imageSearcherOptions =
      [[TFLImageSearcherOptions alloc] initWithModelPath:self.modelPath];

  TFLImageSearcher *imageSearcher = [TFLImageSearcher imageSearcherWithOptions:imageSearcherOptions
                                                                         error:nil];
  XCTAssertNotNil(imageSearcher);

  return imageSearcher;
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

- (void)testSuccessfullInferenceWithSearchContentOnMLImageWithUIImage {
  TFLImageSearcher *imageSearcher =
      [self testSuccessfulCreationOfImageSearcherWithSearchContent:self.modelPath];
  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"burger" ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  TFLSearchResult *searchResult = [imageSearcher searchWithGMLImage:gmlImage error:nil];
  [self verifySearchResultForInferenceWithSearchContent:searchResult];
}

@end

NS_ASSUME_NONNULL_END
