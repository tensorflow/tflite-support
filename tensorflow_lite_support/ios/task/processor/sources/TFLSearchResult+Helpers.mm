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
#import "tensorflow_lite_support/ios/sources/TFLCommonCppUtils.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLSearchResult+Helpers.h"

namespace {
using tflite::support::StatusOr;
using SearchResultCpp = tflite::task::processor::SearchResult;
}

@implementation TFLSearchResult (Helpers)

+ (nullable TFLSearchResult *)searchResultWithCppResult:
                                  (const StatusOr<SearchResultCpp> &)cppSearchResult
                                                  error:(NSError **)error {
  if (![TFLCommonCppUtils checkCppError:cppSearchResult.status() toError:error]) {
    return nil;
  }

  NSMutableArray *nearestNeighbors = [[NSMutableArray alloc] init];

  auto cpp_search_result_value = cppSearchResult.value();
  
  for (int i = 0; i < cpp_search_result_value.nearest_neighbors_size(); i++) {
    auto cpp_nearest_neighbor = cpp_search_result_value.nearest_neighbors(i);
    NSString *metadata = [NSString stringWithCString:cpp_nearest_neighbor.metadata().c_str()
                                            encoding:NSUTF8StringEncoding];
    TFLNearestNeighbor *nearestNeighbor =
        [[TFLNearestNeighbor alloc] initWithMetaData:metadata
                                            distance:(CGFloat)cpp_nearest_neighbor.distance()];
    [nearestNeighbors addObject:nearestNeighbor];
  }

  return [[TFLSearchResult alloc] initWithNearestNeighbors:nearestNeighbors];
}

@end
