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
#import "tensorflow_lite_support/ios/task/vision/sources/TFLImageSearcher.h"
#import "tensorflow_lite_support/ios/sources/TFLCommon.h"
#import "tensorflow_lite_support/ios/sources/TFLCommonUtils.h"
#import "tensorflow_lite_support/ios/task/core/sources/TFLBaseOptions+CppHelpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLEmbeddingOptions+Helpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLSearchOptions+Helpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLSearchResult+Helpers.h"
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImage+CppUtils.h"

#include "tensorflow_lite_support/cc/task/vision/image_searcher.h"

namespace {
using ImageSearcherCpp = ::tflite::task::vision::ImageSearcher;
using ImageSearcherOptionsCpp =
    ::tflite::task::vision::ImageSearcherOptions;
using FrameBufferCpp = ::tflite::task::vision::FrameBuffer;
using BoundingBoxCpp = ::tflite::task::vision::BoundingBox;
using SearchResultCpp = ::tflite::task::processor::SearchResult;
using ::tflite::support::StatusOr;
}

@interface TFLImageSearcher () {
/** ImageSearcher backed by C API */
std::unique_ptr<ImageSearcherCpp> _cppImageSearcher;
}
@end

@implementation TFLImageSearcherOptions
@synthesize baseOptions;
@synthesize embeddingOptions;
@synthesize searchOptions;

- (instancetype)init {
  self = [super init];
  if (self) {
    self.baseOptions = [[TFLBaseOptions alloc] init];
    self.embeddingOptions = [[TFLEmbeddingOptions alloc] init];
    self.searchOptions = [[TFLSearchOptions alloc] init];
  }
  return self;
}

- (instancetype)initWithModelPath:(NSString *)modelPath {
  self = [self init];
  if (self) {
    self.baseOptions.modelFile.filePath = modelPath;
  }
  return self;
}

- (ImageSearcherOptionsCpp)cppOptions {
    ImageSearcherOptionsCpp cppOptions = {};
    [self.baseOptions copyToCppOptions:cppOptions.mutable_base_options()];
    [self.embeddingOptions copyToCppOptions:cppOptions.mutable_embedding_options()];
    [self.searchOptions copyToCppOptions:cppOptions.mutable_search_options()];

    return cppOptions;
}

@end

@implementation TFLImageSearcher

- (nullable instancetype)initWithCppImageSearcherOptions:(ImageSearcherOptionsCpp)cppOptions {
  self = [super init];
  if (self) {
     StatusOr<std::unique_ptr<ImageSearcherCpp>> cppImageSearcher =
      ImageSearcherCpp::CreateFromOptions(cppOptions);
    if (cppImageSearcher.ok()) {
      _cppImageSearcher = std::move(cppImageSearcher.value());
     }
     else {
       return nil;
     }
  }
  return self;
}

+ (nullable instancetype)imageSearcherWithOptions:(TFLImageSearcherOptions *)options
                                              error:(NSError **)error {
  if (!options) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"TFLImageSearcherOptions argument cannot be nil."];
    return nil;
  }
  
  ImageSearcherOptionsCpp cppOptions = [options cppOptions];

  return [[TFLImageSearcher alloc] initWithCppImageSearcherOptions:cppOptions];
}

- (nullable TFLSearchResult *)searchInGMLImage:(GMLImage *)image
                                                     error:(NSError **)error {

  if (!image) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"GMLImage argument cannot be nil."];
    return nil;
  }

  std::unique_ptr<FrameBufferCpp> cppFrameBuffer = [image cppFrameBufferWithError:error];

  if (!cppFrameBuffer) {
    return nil;
  }

  StatusOr<SearchResultCpp> cpp_search_result_status =
      _cppImageSearcher->Search(*cppFrameBuffer);
 
  return [TFLSearchResult searchResultWithCppResult:cpp_search_result_status error:error];
}

- (nullable TFLSearchResult *)searchInGMLImage:(GMLImage *)image
                                          regionOfInterest:(CGRect)roi
                                                     error:(NSError **)error {
  if (!image) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"GMLImage argument cannot be nil."];
    return nil;
  }

  std::unique_ptr<FrameBufferCpp> cppFrameBuffer = [image cppFrameBufferWithError:error];

  if (!cppFrameBuffer) {
    return nil;
  }

  BoundingBoxCpp cc_roi;
  cc_roi.set_origin_x(roi.origin.x);
  cc_roi.set_origin_y(roi.origin.y);
  cc_roi.set_width(roi.size.width);
  cc_roi.set_height(roi.size.height);                            
  
  StatusOr<SearchResultCpp> cpp_search_result_status =
      _cppImageSearcher->Search(*cppFrameBuffer, cc_roi);
 
  return [TFLSearchResult searchResultWithCppResult:cpp_search_result_status error:error];
}
@end
