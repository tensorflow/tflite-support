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
#include "tensorflow_lite_support/cc/task/processor/proto/search_result_proto_inc.h"
#include "tensorflow_lite_support/cc/port/statusor.h"

#import "tensorflow_lite_support/ios/task/processor/sources/TFLSearchResult.h"

NS_ASSUME_NONNULL_BEGIN

@interface TFLSearchResult (Helpers)
/**
 * Creates and retrurns a TFLSegmentationResult from a TFLSegmentationResult
 * returned by TFLite Task C Library Image Segmentation task.
 *
 * @param cppSearchResult Segmentation results returned by TFLite Task
 * C Library Image Segmentation task.
 *
 * @return Segmentation Result of type TFLSegmentationResult to be returned by
 * inference methods of the iOS TF Lite Task Image Segmentation task.
 */
+ (nullable TFLSearchResult *)searchResultWithCppResult:
    (const tflite::support::StatusOr<tflite::task::processor::SearchResult>&)cppSearchResult error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
