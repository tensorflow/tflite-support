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
#import "tensorflow_lite_support/ios/utils/sources/TFLCommonUtils.h"
#import "tensorflow_lite_support/ios/task/core/sources/TFLBaseOptions+CppHelpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLEmbeddingOptions+Helpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLSearchOptions+Helpers.h"

// #import "tensorflow_lite_support/ios/task/processor/sources/TFLClassificationOptions+Helpers.h"
// #import "tensorflow_lite_support/ios/task/processor/sources/TFLClassificationResult+Helpers.h"
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImage+Utils.h"

#include "tensorflow_lite_support/cc/task/vision/image_searcher.h"

namespace {
using ImageSearcherCpp = ::tflite::task::vision::ImageSearcher;
using ImageSearcherOptionsCpp =
    ::tflite::task::vision::ImageSearcherOptions;
using ::tflite::support::StatusOr;
}

@interface TFLImageSearcher () {
/** ImageClassifier backed by C API */
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
    [self.baseOptions copyTocppOptions:cppOptions.mutable_base_options()];
    [self.embeddingOptions copyToCppOptions:cppOptions.mutable_embedding_options()];
    [self.searchOptions copyToCppOptions:cppOptions.mutable_search_options()];

    return cppOptions;
}

@end

@implementation TFLImageSearcher
// - (void)dealloc {
//   TfLiteImageClassifierDelete(_imageClassifier);
// }

// - (ImageSearcherOptionsCpp) imageSearcherCppOptionsWihOptions(TFLImageSearcherOptions *)options {

//   // if (options == nullptr) {
//   //   return CreateStatusWithPayload(
//   //       absl::StatusCode::kInvalidArgument,
//   //       absl::StrFormat("Expected non null options."),
//   //       TfLiteSupportStatus::kInvalidArgumentError);
//   // }

//   ImageSearcherOptionsCpp cppOptions = {};


//   // More file sources can be added in else ifs
//   if (options.baseOptions.modelFile.filePath) {
//     cppOptions.mutable_base_options()->mutable_model_file()->set_file_name(
//         options.baseOptions.modelFile.filePath);
//   }

//   // c_options->base_options.compute_settings.num_threads is expected to be
//   // set to value > 0 or -1. Otherwise invoking
//   // ImageClassifierCpp::CreateFromOptions() results in a not ok status.
//   cppOptions.mutable_base_options()
//       ->mutable_compute_settings()
//       ->mutable_tflite_settings()
//       ->mutable_cpu_settings()
//       ->set_num_threads(
//           options->baseOptions.computeSettings.cpuSettings.numThreads);

//   // for (int i = 0; i < c_options->classification_options.label_denylist.length;
//   //      i++)
//   //   cpp_options.add_class_name_blacklist(
//   //       c_options->classification_options.label_denylist.list[i]);

//   // for (int i = 0; i < c_options->classification_options.label_allowlist.length;
//   //      i++)
//   //   cpp_options.add_class_name_whitelist(
//   //       c_options->classification_options.label_allowlist.list[i]);

//   // // Check needed since setting a nullptr for this field results in a segfault
//   // // on invocation of ImageClassifierCpp::CreateFromOptions().
//   // if (c_options->classification_options.display_names_local) {
//   //   cpp_options.set_display_names_locale(
//   //       c_options->classification_options.display_names_local);
//   // }

//   // c_options->classification_options.max_results is expected to be set to -1
//   // or any value > 0. Otherwise invoking
//   // ImageClassifierCpp::CreateFromOptions() results in a not ok status.
//   cpp_options.set_max_results(c_options->classification_options.max_results);

//   cpp_options.set_score_threshold(
//       c_options->classification_options.score_threshold);

//   return cpp_options;
// }

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
  return [self searchInGMLImage:image
                   regionOfInterest:CGRectMake(0, 0, image.width, image.height)
                              error:error];
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

  // TfLiteBoundingBox boundingBox = {.origin_x = roi.origin.x,
  //                                  .origin_y = roi.origin.y,
  //                                  .width = roi.size.width,
  //                                  .height = roi.size.height};

  BoundingBoxCpp cc_roi;
  if (roi == nullptr) {
    cc_roi.set_width(frame_buffer->dimension.width);
    cc_roi.set_height(frame_buffer->dimension.height);
  } else {
    cc_roi.set_origin_x(roi->origin_x);
    cc_roi.set_origin_y(roi->origin_y);
    cc_roi.set_width(roi->width);
    cc_roi.set_height(roi->height);
  }                             
  
  StatusOr<SearchResultCpp> cpp_search_result_status =
      _cppImageSearcher->Search(cppFrameBuffer, cc_roi);

  if (!cpp_classification_result_status.ok()) {
    tflite::support::CreateTfLiteSupportErrorWithStatus(
        cpp_classification_result_status.status(), error);
    return nullptr;

  // TfLiteSupportError *classifyError = NULL;
  // TfLiteClassificationResult *cClassificationResult = TfLiteImageClassifierClassifyWithRoi(
  //     _imageClassifier, cFrameBuffer, &boundingBox, &classifyError);

  free(cFrameBuffer->buffer);
  cFrameBuffer->buffer = NULL;

  free(cFrameBuffer);
  cFrameBuffer = NULL;

  // Populate iOS error if C Error is not null and afterwards delete it.
  if (![TFLCommonUtils checkCError:classifyError toError:error]) {
    TfLiteSupportErrorDelete(classifyError);
  }

  // Return nil if C result evaluates to nil. If an error was generted by the C layer, it has
  // already been populated to an NSError and deleted before returning from the method.
  if (!cClassificationResult) {
    return nil;
  }

  TFLClassificationResult *classificationHeadsResults =
      [TFLClassificationResult classificationResultWithCResult:cClassificationResult];
  TfLiteClassificationResultDelete(cClassificationResult);

  return classificationHeadsResults;
}
@end
