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
#import "tensorflow_lite_support/ios/task/audio/sources/TFLAudioClassifier.h"
#import "tensorflow_lite_support/ios/sources/TFLCommon.h"
#import "tensorflow_lite_support/ios/sources/TFLCommonUtils.h"
#import "tensorflow_lite_support/ios/task/audio/core/audio_tensor/utils/sources/TFLAudioTensor+Utils.h"
#import "tensorflow_lite_support/ios/task/core/sources/TFLBaseOptions+Helpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLClassificationOptions+Helpers.h"
#import "tensorflow_lite_support/ios/task/processor/sources/TFLClassificationResult+Helpers.h"

#include "tensorflow_lite_support/c/task/audio/audio_classifier.h"

@interface TFLAudioClassifier () {
  NSInteger _requiredCBufferSize;
  TfLiteAudioFormat *_requiredCAudioFormat;
}
/** Audio Classifier backed by C API */
@property(nonatomic) TfLiteAudioClassifier *audioClassifier;
@end

@implementation TFLAudioClassifierOptions

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

- (instancetype)initWithModelPath:(NSString *)modelPath {
  self = [self init];
  if (self) {
    self.baseOptions.modelFile.filePath = modelPath;
  }
  return self;
}

@end

@implementation TFLAudioClassifier
- (void)dealloc {
  TfLiteAudioFormatDelete(_requiredCAudioFormat);
  TfLiteAudioClassifierDelete(_audioClassifier);
}

- (instancetype)initWithAudioClassifier:(TfLiteAudioClassifier *)audioClassifier {
  self = [super init];
  if (self) {
    _audioClassifier = audioClassifier;
  }
  return self;
}

- (nullable instancetype)initWithCOptions:(TfLiteAudioClassifierOptions)cOptions
                                    error:(NSError **)error {
  TfLiteSupportError *cCreateClassifierError = NULL;
  TfLiteAudioClassifier *cAudioClassifier =
      TfLiteAudioClassifierFromOptions(&cOptions, &cCreateClassifierError);

  // Populate iOS error if TfliteSupportError is not null and afterwards delete it.
  if (![TFLCommonUtils checkCError:cCreateClassifierError toError:error]) {
    TfLiteSupportErrorDelete(cCreateClassifierError);
  }

  // Return nil if classifier evaluates to nil. If an error was generted by the C layer, it has
  // already been populated to an NSError and deleted before returning from the method.
  if (!cAudioClassifier) {
    return nil;
  }

  return [self initWithAudioClassifier:cAudioClassifier];
}

- (nullable instancetype)initWithOptions:(TFLAudioClassifierOptions *)options
                                   error:(NSError **)error {
  if (!options) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"modelPath argument cannot be nil."];
    return nil;
  }

  TfLiteAudioClassifierOptions cOptions = TfLiteAudioClassifierOptionsCreate();

  if (![options.classificationOptions copyToCOptions:&(cOptions.classification_options)
                                               error:error]) {
    [options.classificationOptions
        deleteAllocatedMemoryOfClassificationOptions:&(cOptions.classification_options)];
    return nil;
  }

  [options.baseOptions copyToCOptions:&(cOptions.base_options)];

  TFLAudioClassifier *audioClassifier = [self initWithCOptions:cOptions error:error];

  [options.classificationOptions
      deleteAllocatedMemoryOfClassificationOptions:&(cOptions.classification_options)];
  
  if(![self populateRequiredBufferSizeWithError:error]) {
    NSLog(@"Check bool");
    return nil;
  } 

  if(![self populateRequiredTensorFormatWithError:error]) {
        NSLog(@"Check bool 1");
    return nil;
  } 

  return audioClassifier;
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath error:(NSError **)error {
  TFLAudioClassifierOptions *options =
      [[TFLAudioClassifierOptions alloc] initWithModelPath:modelPath];

  return [self initWithOptions:options error:error];
}

- (nullable TFLClassificationResult *)classifyWithAudioTensor:(TFLAudioTensor *)audioTensor
                                                        error:(NSError **)error {
  if (!audioTensor) {
    [TFLCommonUtils createCustomError:error
                             withCode:TFLSupportErrorCodeInvalidArgumentError
                          description:@"audioTensor argument cannot be nil."];
    return nil;
  }

  TfLiteAudioBuffer cAudioBuffer = [audioTensor cAudioBufferFromFloatBuffer:audioTensor.buffer];

  TfLiteSupportError *classifyError = NULL;
  TfLiteClassificationResult *cClassificationResult =
      TfLiteAudioClassifierClassify(_audioClassifier, &cAudioBuffer, &classifyError);

  // Populate iOS error if C Error is not null and afterwards delete it.
  if (![TFLCommonUtils checkCError:classifyError toError:error]) {
    TfLiteSupportErrorDelete(classifyError);
  }

  // Return nil if C result evaluates to nil. If an error was generted by the C layer, it has
  // already been populated to an NSError and deleted before returning from the method.
  if (!cClassificationResult) {
    return nil;
  }

  TFLClassificationResult *classificationResult =
      [TFLClassificationResult classificationResultWithCResult:cClassificationResult];
  TfLiteClassificationResultDelete(cClassificationResult);

  return classificationResult;
}

- (BOOL)populateRequiredBufferSizeWithError:(NSError **)error {
  TfLiteSupportError *requiredBufferSizeError = NULL;
  _requiredCBufferSize = TfLiteAudioClassifierGetRequiredInputBufferSize(_audioClassifier, &requiredBufferSizeError);

  // Populate iOS error if C Error is not null and afterwards delete it.
  if (![TFLCommonUtils checkCError:requiredBufferSizeError toError:error]) {
    TfLiteSupportErrorDelete(requiredBufferSizeError);
  }
  
  NSLog(@"%d Bufferr", _requiredCBufferSize);
  return _requiredCBufferSize > 0;
}

- (BOOL)populateRequiredTensorFormatWithError:(NSError **)error {
  NSLog(@"check pop tensor");
  TfLiteSupportError *getAudioFormatError = nil;
  _requiredCAudioFormat =
      TfLiteAudioClassifierGetRequiredAudioFormat(_audioClassifier, &getAudioFormatError);

  if (![TFLCommonUtils checkCError:getAudioFormatError toError:error]) {
    TfLiteSupportErrorDelete(getAudioFormatError);
  }
  
  NSLog(@"check poop tensor after");
  return _requiredCAudioFormat != NULL;

  // if (!cFormat) {
  //   return NO;
  // }

  // requiredAudioFormat = [[TFLAudioFormat alloc] initWithChannelCount:cFormat->channels
  //                                                            sampleRate:cFormat->sample_rate];

  // TfLiteAudioFormatDelete(cFormat);

  // return YES;
}

- (TFLAudioTensor *)createInputAudioTensor {

  TFLAudioFormat *format = [[TFLAudioFormat alloc] initWithChannelCount:_requiredCAudioFormat->channels
                                                                  sampleRate:_requiredCAudioFormat->sample_rate];
  return [[TFLAudioTensor alloc] initWithAudioFormat:format
                                         sampleCount:_requiredCBufferSize / format.channelCount];
}

- (TFLAudioRecord *)createAudioRecordWithError:(NSError **)error {

  // The sample count of audio record should be strictly longer than audio tensor's so that
  // clients could run TFLAudioRecord's `startRecordingWithError:`
  // together with TFLAudioClassifier's `classifyWithAudioTensor:error:`
  TFLAudioFormat *format = [[TFLAudioFormat alloc] initWithChannelCount:_requiredCAudioFormat->channels
                                                                  sampleRate:_requiredCAudioFormat->sample_rate];

  return [[TFLAudioRecord alloc] initWithAudioFormat:format bufferSize:_requiredCBufferSize * 2 error:error];
}

@end
