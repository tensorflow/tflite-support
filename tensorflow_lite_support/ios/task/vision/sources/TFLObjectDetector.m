//
//  TFLObjectDetector.m
//  ObjectDetection
//
//  Created by Prianka Kariat on 13/12/21.
//

#import "tensorflow_lite_support/ios/task/vision/sources/TFLObjectDetector.h"
#import "tensorflow_lite_support/ios/sources/TFLCommon.h"
#import "tensorflow_lite_support/ios/sources/TFLCommonUtils.h"
#import "tensorflow_lite_support/ios/task/core/sources/TFLBaseOptions+Helpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLClassificationOptions+Helpers.h"
#import "tensorflow_lite_support/ios/task/processor/utils/sources/TFLDetectionUtils.h"
#import "tensorflow_lite_support/ios/task/vision/utils/sources/GMLImageUtils.h"

#include "tensorflow_lite_support/c/task/vision/object_detector.h"

@interface TFLObjectDetector ()
/** ObjectDetector backed by C API */
@property(nonatomic) TfLiteObjectDetector *objectDetector;
@end

@implementation TFLObjectDetectorOptions
@synthesize baseOptions;
@synthesize classificationOptions;

- (instancetype)init {
  self = [super init];
  if (self) {
    self.baseOptions = [[TFLBaseOptions alloc] init];
    self.classificationOptions = [[TFLClassificationOptions alloc] init];
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

@implementation TFLObjectDetector
- (void)dealloc {
  TfLiteObjectDetectorDelete(_objectDetector);
}

- (instancetype)initWithObjectDetector:(TfLiteObjectDetector *)objectDetector {
  self = [super init];
  if (self) {
    _objectDetector = objectDetector;
  }
  return self;
}

+ (nullable instancetype)objectDetectorWithOptions:(nonnull TFLObjectDetectorOptions *)options
                                             error:(NSError **)error {
  TfLiteObjectDetectorOptions cOptions = TfLiteObjectDetectorOptionsCreate();
  if (![options.classificationOptions
          copyClassificationOptionsToCClassificationOptions:&(cOptions.classification_options)
                                                      error:error])
    return nil;

  [options.baseOptions copyBaseOptionsToCBaseOptions:&(cOptions.base_options)];

  TfLiteSupportError *createObjectDetectorError = nil;
  TfLiteObjectDetector *objectDetector =
      TfLiteObjectDetectorFromOptions(&cOptions, &createObjectDetectorError);

  [options.classificationOptions
      deleteCStringArraysOfClassificationOptions:&(cOptions.classification_options)];

  if (!objectDetector) {
    [TFLCommonUtils errorFromTfLiteSupportError:createObjectDetectorError error:error];
    TfLiteSupportErrorDelete(createObjectDetectorError);
    return nil;
  }

  return [[TFLObjectDetector alloc] initWithObjectDetector:objectDetector];
}

- (nullable TFLDetectionResult *)detectWithGMLImage:(GMLImage *)image
                                              error:(NSError *_Nullable *)error {
  TfLiteFrameBuffer *cFrameBuffer = [GMLImageUtils cFrameBufferFromGMLImage:image error:error];

  if (!cFrameBuffer) {
    return nil;
  }

  TfLiteSupportError *detectError = nil;
  TfLiteDetectionResult *cDetectionResult =
      TfLiteObjectDetectorDetect(_objectDetector, cFrameBuffer, &detectError);

  free(cFrameBuffer->buffer);
  cFrameBuffer->buffer = nil;

  free(cFrameBuffer);
  cFrameBuffer = nil;

  if (!cDetectionResult) {
    [TFLCommonUtils errorFromTfLiteSupportError:detectError error:error];
    TfLiteSupportErrorDelete(detectError);
    return nil;
  }

  TFLDetectionResult *detectionResult =
      [TFLDetectionUtils detectionResultFromCDetectionResults:cDetectionResult];
  TfLiteDetectionResultDelete(cDetectionResult);

  return detectionResult;
}

@end
