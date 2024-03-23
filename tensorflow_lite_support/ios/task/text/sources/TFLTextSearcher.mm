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
#import "tensorflow_lite_support/ios/task/text/sources/TFLTextSearcher.h"
#import "tensorflow_lite_support/ios/sources/TFLCommon.h"
#import "tensorflow_lite_support/ios/sources/TFLCommonUtils.h"
#import "tensorflow_lite_support/ios/task/core/sources/TFLBaseOptions+CppHelpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLEmbeddingOptions+Helpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLSearchOptions+Helpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLSearchResult+Helpers.h"

#include "tensorflow_lite_support/cc/task/text/text_searcher.h"

namespace {
using TextSearcherCpp = ::tflite::task::text::TextSearcher;
using TextSearcherOptionsCpp = ::tflite::task::text::TextSearcherOptions;
using SearchResultCpp = ::tflite::task::processor::SearchResult;
using ::tflite::support::StatusOr;
}  // namespace

@interface TFLTextSearcher () {
  /** TextSearcher backed by C++ API */
  std::unique_ptr<TextSearcherCpp> _cppTextSearcher;
}
@end

@implementation TFLTextSearcherOptions

- (instancetype)init {
  self = [super init];
  if (self) {
    _baseOptions = [[TFLBaseOptions alloc] init];
    _embeddingOptions = [[TFLEmbeddingOptions alloc] init];
    _searchOptions = [[TFLSearchOptions alloc] init];
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath {
  self = [self init];
  if (self) {
    _baseOptions.modelFile.filePath = modelPath;
  }
  return self;
}

- (TextSearcherOptionsCpp)cppOptions {
  TextSearcherOptionsCpp cppOptions = {};
  [self.baseOptions copyToCppOptions:cppOptions.mutable_base_options()];
  [self.embeddingOptions copyToCppOptions:cppOptions.mutable_embedding_options()];
  [self.searchOptions copyToCppOptions:cppOptions.mutable_search_options()];

  return cppOptions;
}

@end

@implementation TFLTextSearcher

- (nullable instancetype)initWithCppTextSearcherOptions:(TextSearcherOptionsCpp)cppOptions {
  self = [super init];
  if (self) {
    StatusOr<std::unique_ptr<TextSearcherCpp>> cppTextSearcher =
        TextSearcherCpp::CreateFromOptions(cppOptions);
    if (cppTextSearcher.ok()) {
      _cppTextSearcher = std::move(cppTextSearcher.value());
    } else {
      return nil;
    }
  }
  return self;
}

+ (nullable instancetype)textSearcherWithOptions:(TFLTextSearcherOptions *)options
                                            error:(NSError **)error {
  if (!options) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"TFLTextSearcherOptions argument cannot be nil."];
    return nil;
  }

  TextSearcherOptionsCpp cppOptions = [options cppOptions];

  return [[TFLTextSearcher alloc] initWithCppTextSearcherOptions:cppOptions];
}

- (nullable TFLSearchResult *)searchWithText:(NSString *)text error:(NSError **)error {
  if (!text) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"GMLImage argument cannot be nil."];
    return nil;
  }

  std::string cppTextToBeSearched =  std::string(text.UTF8String, [text lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
  StatusOr<SearchResultCpp> cppSearchResultStatus = _cppTextSearcher->Search(
    cppTextToBeSearched);

  return [TFLSearchResult searchResultWithCppResult:cppSearchResultStatus error:error];
}

@end
