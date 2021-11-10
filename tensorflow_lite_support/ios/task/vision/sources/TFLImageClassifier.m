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
#import "tensorflow_lite_support/ios/task/vision/sources/TFLImageClassifier.h"
#import "tensorflow_lite_support/ios/sources/TFLCommon.h"
#import "tensorflow_lite_support/ios/sources/TFLCommonUtils.h"
#import "tensorflow_lite_support/ios/task/processor/utils/sources/TFLClassificationUtils.h"
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImageUtils.h"

#include "tensorflow_lite_support/c/task/vision/image_classifier.h"

@interface TFLImageClassifier ()
/** BertNLClassifier backed by C API */
@property(nonatomic) TfLiteImageClassifier *imageClassifier;
@end

@implementation TFLImageClassifierOptions
@synthesize baseOptions;
@synthesize classificationOptions;

- (instancetype)init {
  self = [super init];
  if (self) {
    self.baseOptions = [[TFLBaseOptions alloc] init];
    self.baseOptions.modelFile = [[TFLExternalFile alloc] init];

    self.baseOptions.computeSettings = [[TFLComputeSettings alloc] init];
    self.baseOptions.computeSettings.cpuSettings = [[TFLCpuSettings alloc] init];
    self.baseOptions.computeSettings.cpuSettings.numThreads = -1;

    self.classificationOptions = [[TFLClassificationOptions alloc] init];
    self.classificationOptions.maxResults = 5;
    self.classificationOptions.scoreThreshold = 0;
  }
  return self;
}

- (nullable instancetype)initWithModelPath:(nonnull NSString *)modelPath {
  self = [self init];
  if (self) {
    self.baseOptions.modelFile.filePath = modelPath;
  }
  return self;
}

@end

@implementation TFLImageClassifier
- (void)dealloc {
  TfLiteImageClassifierDelete(_imageClassifier);
}

+ (void)deleteCStringsArray:(char **)cStrings count:(int)count {
  for (NSInteger i = 0; i < count; i++) {
    free(cStrings[i]);
  }

  free(cStrings);
}

+ (char **)cStringArrayFromNSArray:(NSArray<NSString *> *)strings error:(NSError **)error {
  if (strings.count <= 0) {
    [TFLCommonUtils customErrorWithCode:TFLSupportErrorCodeInvalidArgumentError
                            description:@"Invalid length of strings found for list type options."
                                  error:error];
    return NULL;
  }

  char **cStrings = (char **)calloc(strings.count, sizeof(char *));

  if (!cStrings) {
    [TFLCommonUtils customErrorWithCode:TFLSupportErrorCodeInternalError
                            description:@"Could not initialize list type options."
                                  error:error];
    return nil;
  }

  for (NSInteger i = 0; i < strings.count; i++) {
    char *cString = [TFLCommonUtils
        mallocWithSize:[strings[i] lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1
                 error:error];
    if (!cString) return nil;

    strcpy(cString, strings[i].UTF8String);
  }

  return cStrings;
}

- (instancetype)initWithImageClassifier:(TfLiteImageClassifier *)imageClassifier {
  self = [super init];
  if (self) {
    _imageClassifier = imageClassifier;
  }
  return self;
}

+ (nullable instancetype)imageClassifierWithOptions:(nonnull TFLImageClassifierOptions *)options
                                              error:(NSError **)error {
  TfLiteImageClassifierOptions cOptions = TfLiteImageClassifierOptionsCreate();

  cOptions.classification_options.score_threshold = options.classificationOptions.scoreThreshold;
  cOptions.classification_options.max_results = (int)options.classificationOptions.maxResults;
  cOptions.base_options.compute_settings.cpu_settings.num_threads =
      (int)options.baseOptions.computeSettings.cpuSettings.numThreads;

  if (options.classificationOptions.labelDenyList) {
    char **cClassNameBlackList =
        [TFLImageClassifier cStringArrayFromNSArray:options.classificationOptions.labelDenyList
                                              error:error];
    if (!cClassNameBlackList) {
      return nil;
    }

    cOptions.classification_options.label_denylist.list = cClassNameBlackList;
    cOptions.classification_options.label_denylist.length =
        (int)options.classificationOptions.labelDenyList.count;
  }

  if (options.classificationOptions.labelAllowList) {
    char **cClassNameWhiteList =
        [TFLImageClassifier cStringArrayFromNSArray:options.classificationOptions.labelAllowList
                                              error:error];
    if (!cClassNameWhiteList) {
      return nil;
    }

    cOptions.classification_options.label_allowlist.list = cClassNameWhiteList;
    cOptions.classification_options.label_allowlist.length =
        (int)options.classificationOptions.labelAllowList.count;
  }

  if (options.classificationOptions.displayNamesLocal) {
    cOptions.classification_options.display_names_local =
        (char *)options.classificationOptions.displayNamesLocal.UTF8String;
  }

  if (options.baseOptions.modelFile.filePath) {
    cOptions.base_options.model_file.file_path = options.baseOptions.modelFile.filePath.UTF8String;
  }

  TfLiteSupportError *createClassifierError = nil;
  TfLiteImageClassifier *imageClassifier =
      TfLiteImageClassifierFromOptions(&cOptions, &createClassifierError);

  if (options.classificationOptions.labelAllowList) {
    [TFLImageClassifier deleteCStringsArray:cOptions.classification_options.label_allowlist.list
                                      count:cOptions.classification_options.label_allowlist.length];
  }

  if (options.classificationOptions.labelDenyList) {
    [TFLImageClassifier deleteCStringsArray:cOptions.classification_options.label_denylist.list
                                      count:cOptions.classification_options.label_denylist.length];
  }

  if (!imageClassifier) {
    [TFLCommonUtils errorFromTfLiteSupportError:createClassifierError error:error];
    TfLiteSupportErrorDelete(createClassifierError);
    return nil;
  }

  return [[TFLImageClassifier alloc] initWithImageClassifier:imageClassifier];
}

+ (nullable instancetype)imageClassifierWithModelPath:(nonnull NSString *)modelPath
                                                error:(NSError **)error {
  TFLImageClassifierOptions *options = [[TFLImageClassifierOptions alloc] init];

  TFLImageClassifier *imageClassifier = nil;
  if (options) {
    options.baseOptions.modelFile.filePath = modelPath;
    imageClassifier = [TFLImageClassifier imageClassifierWithOptions:options error:error];
  } else
    [TFLCommonUtils
        customErrorWithCode:TFLSupportErrorCodeInternalError
                description:@"Some error occured during initialization of image classifier."
                      error:error];

  return imageClassifier;
}

- (nullable TFLClassificationResult *)classifyWithGMLImage:(GMLImage *)image
                                                     error:(NSError *_Nullable *)error {
  return [self classifyWithGMLImage:image
                   regionOfInterest:CGRectMake(0, 0, image.width, image.height)
                              error:error];
}

- (nullable TFLClassificationResult *)classifyWithGMLImage:(GMLImage *)image
                                          regionOfInterest:(CGRect)roi
                                                     error:(NSError *_Nullable *)error {
  TfLiteFrameBuffer *cFrameBuffer = [GMLImageUtils cFrameBufferFromGMLImage:image error:error];

  if (!cFrameBuffer) {
    return nil;
  }

  TfLiteBoundingBox boundingBox = {.origin_x = roi.origin.x,
                                   .origin_y = roi.origin.y,
                                   .width = roi.size.width,
                                   .height = roi.size.height};

  TfLiteSupportError *classifyError = nil;
  TfLiteClassificationResult *cClassificationResult = TfLiteImageClassifierClassifyWithRoi(
      _imageClassifier, cFrameBuffer, &boundingBox, &classifyError);

  free(cFrameBuffer->buffer);
  cFrameBuffer->buffer = nil;

  free(cFrameBuffer);
  cFrameBuffer = nil;

  if (!cClassificationResult) {
    [TFLCommonUtils errorFromTfLiteSupportError:classifyError error:error];
    TfLiteSupportErrorDelete(classifyError);
    return nil;
  }

  TFLClassificationResult *classificationHeadsResults =
      [TFLClassificationUtils classificationResultFromCClassificationResults:cClassificationResult];
  TfLiteClassificationResultDelete(cClassificationResult);

  return classificationHeadsResults;
}
@end
