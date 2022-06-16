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

#import "tensorflow_lite_support/ios/task/audio/sources/TFLAudioClassifier.h"
#import "tensorflow_lite_support/ios/test/task/audio/core/audio_record/utils/sources/AVAudioPCMBuffer+Utils.h"

#define VerifyError(error, expectedDomain, expectedCode, expectedLocalizedDescription)  \
  XCTAssertNotNil(error);                                                               \
  XCTAssertEqual(error.domain, expectedDomain);                                         \
  XCTAssertEqual(error.code, expectedCode);                                             \
  XCTAssertNotEqual(                                                                    \
      [error.localizedDescription rangeOfString:expectedLocalizedDescription].location, \
      NSNotFound)

#define VerifyCategory(category, expectedIndex, expectedScore, expectedLabel, expectedDisplayName) \
  XCTAssertEqual(category.index, expectedIndex);                                                   \
  XCTAssertEqualWithAccuracy(category.score, expectedScore, 1e-6);                                 \
  XCTAssertEqualObjects(category.label, expectedLabel);                                            \
  XCTAssertEqualObjects(category.displayName, expectedDisplayName);

#define VerifyClassifications(classifications, expectedHeadIndex, expectedCategoryCount) \
  XCTAssertEqual(classifications.categories.count, expectedCategoryCount);               \
  XCTAssertEqual(classifications.headIndex, expectedHeadIndex)

#define VerifyClassificationResult(classificationResult, expectedClassificationsCount) \
  XCTAssertNotNil(classificationResult);                                               \
  XCTAssertEqual(classificationResult.classifications.count, expectedClassificationsCount)

static NSString *const expectedTaskErrorDomain = @"org.tensorflow.lite.tasks";

NS_ASSUME_NONNULL_BEGIN

@interface TFLAudioClassifierTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@property(nonatomic) AVAudioFormat *audioEngineFormat;
@end

// This category of TFLAudioRecord is private to the current test file. This is needed in order to
// expose the method to load the audio record buffer without calling: -[TFLAudioRecord
// startRecordingWithError:]. This is needed to avoid exposing this method which isn't useful to the
// consumers of the framework.
@interface TFLAudioRecord (Tests)
- (void)convertAndLoadBuffer:(AVAudioPCMBuffer *)buffer
         usingAudioConverter:(AVAudioConverter *)audioConverter;
@end

@implementation TFLAudioClassifierTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class.
  [super setUp];
  self.modelPath =
      [[NSBundle bundleForClass:self.class] pathForResource:@"yamnet_audio_classifier_with_metadata"
                                                     ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);

  self.audioEngineFormat = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                                            sampleRate:48000
                                                              channels:1
                                                           interleaved:NO];
}

- (nullable AVAudioPCMBuffer *)bufferFromFileWithName:(NSString *)name
                                            extension:(NSString *)extension
                                          audioFormat:(TFLAudioFormat *)audioFormat {
  NSString *filePath = [[NSBundle bundleForClass:self.class] pathForResource:name ofType:extension];

  return [AVAudioPCMBuffer loadPCMBufferFromFileWithPath:filePath audioFormat:audioFormat];
}

- (nullable AVAudioPCMBuffer *)bufferFromFileWithName:(NSString *)name
                                            extension:(NSString *)extension
                                     processingFormat:(AVAudioFormat *)processingFormat {
  NSString *filePath = [[NSBundle bundleForClass:self.class] pathForResource:name ofType:extension];

  return [AVAudioPCMBuffer loadPCMBufferFromFileWithPath:filePath
                                        processingFormat:processingFormat];
}

- (void)testInferenceWithFloatBufferSucceeds {
  TFLAudioClassifierOptions *options = [[TFLAudioClassifierOptions alloc] initWithModelPath:self.modelPath];
  TFLAudioClassifier *audioClassifier = [TFLAudioClassifier audioClassifierWithOptions:options
                                                                                 error:nil];
  XCTAssertNotNil(audioClassifier);

  // Create the audio tensor using audio classifier.
  TFLAudioTensor *audioTensor = [audioClassifier createInputAudioTensor];
  XCTAssertNotNil(audioTensor);

  // Load pcm buffer from file.
  AVAudioPCMBuffer *buffer = [self bufferFromFileWithName:@"speech"
                                                extension:@"wav"
                                              audioFormat:audioTensor.audioFormat];

  // Get float buffer from pcm buffer.
  TFLFloatBuffer *floatBuffer = buffer.floatBuffer;
  XCTAssertNotNil(floatBuffer);

  // Load float buffer into the audio tensor.
  [audioTensor loadBuffer:floatBuffer offset:0 size:floatBuffer.size error:nil];

  // Perform classification on audio tensor.
  TFLClassificationResult *classificationResult =
      [audioClassifier classifyWithAudioTensor:audioTensor error:nil];

  const NSInteger expectedClassificationsCount = 1;
  VerifyClassificationResult(classificationResult, expectedClassificationsCount);

  const NSInteger expectedHeadIndex = 0;
  const NSInteger expectedCategoryCount = 521;
  VerifyClassifications(classificationResult.classifications[0], expectedHeadIndex,
                        expectedCategoryCount);
  VerifyCategory(classificationResult.classifications[0].categories[0],
                 0,          // expectedIndex
                 0.957031,   // expectedScore
                 @"Speech",  // expectedLabel
                 nil         // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[1],
                 500,                    // expectedIndex
                 0.019531,               // expectedScore
                 @"Inside, small room",  // expectedLabel
                 nil                     // expectedDisplaName
  );
  // The 3rd result is different from python tests because of the audio file format conversions are
  // done using iOS native classes to mimic audio record behaviour. The iOS native classes handle
  // audio format conversion differently as opposed to the task library C++ convenience method.
  VerifyCategory(classificationResult.classifications[0].categories[2],
                 485,          // expectedIndex
                 0.003906,     // expectedScore
                 @"Clicking",  // expectedLabel
                 nil           // expectedDisplaName
  );
}

- (void)testInferenceWithAudioRecordSucceeds {
  TFLAudioClassifierOptions *options = [[TFLAudioClassifierOptions alloc] initWithModelPath:self.modelPath];
  TFLAudioClassifier *audioClassifier = [TFLAudioClassifier audioClassifierWithOptions:options
                                                                                 error:nil];
  XCTAssertNotNil(audioClassifier);

  // Create audio record using audio classifier
  TFLAudioRecord *audioRecord = [audioClassifier createAudioRecordWithError:nil];

  XCTAssertNotNil(audioRecord);

  // Loading AVAudioPCMBuffer with an array is not currently supported for iOS versions < 15.0.
  // Instead audio samples from a wav file are loaded and converted into the same format
  // of AVAudioEngine's input node to mock the input from the AVAudio Engine.
  AVAudioPCMBuffer *audioEngineBuffer = [self bufferFromFileWithName:@"speech"
                                                           extension:@"wav"
                                                    processingFormat:self.audioEngineFormat];
  XCTAssertNotNil(audioEngineBuffer);

  // Convert the buffer in the audio engine input format to the format with which audio record is
  // intended to output the audio samples. This mocks the internal conversion of audio record when
  // -[TFLAudioRecord startRecording:withError:] is called.
  AVAudioFormat *recordingFormat = [[AVAudioFormat alloc]
      initWithCommonFormat:AVAudioPCMFormatFloat32
                sampleRate:audioRecord.audioFormat.sampleRate
                  channels:(AVAudioChannelCount)audioRecord.audioFormat.channelCount
               interleaved:YES];

  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:self.audioEngineFormat
                                                                     toFormat:recordingFormat];
  // Convert and load the buffer of `TFLAudioRecord`.
  [audioRecord convertAndLoadBuffer:audioEngineBuffer usingAudioConverter:audioConverter];

  // Create audio tensor using audio classifier.
  TFLAudioTensor *audioTensor = [audioClassifier createInputAudioTensor];
  XCTAssertNotNil(audioTensor);

  // Load the audioRecord buffer into the audio tensor.
  [audioTensor loadAudioRecord:audioRecord withError:nil];

  // Perform classification on audio tensor.
  TFLClassificationResult *classificationResult =
      [audioClassifier classifyWithAudioTensor:audioTensor error:nil];

  const NSInteger expectedClassificationsCount = 1;
  VerifyClassificationResult(classificationResult, expectedClassificationsCount);

  const NSInteger expectedHeadIndex = 0;
  const NSInteger expectedCategoryCount = 521;
  VerifyClassifications(classificationResult.classifications[0], expectedHeadIndex,
                        expectedCategoryCount);
  VerifyCategory(classificationResult.classifications[0].categories[0],
                 0,          // expectedIndex
                 0.957031,   // expectedScore
                 @"Speech",  // expectedLabel
                 nil         // expectedDisplaName
  );
  VerifyCategory(classificationResult.classifications[0].categories[1],
                 500,                    // expectedIndex
                 0.019531,               // expectedScore
                 @"Inside, small room",  // expectedLabel
                 nil                     // expectedDisplaName
  );
  // The 3rd result is different from python tests because of the audio file format conversions are
  // done using iOS native classes to mimic audio record behaviour. The iOS native classes handle
  // audio format conversion differently as opposed to the task library C++ convenience method.
  VerifyCategory(classificationResult.classifications[0].categories[2],
                 380,                   // expectedIndex
                 0.003906,              // expectedScore
                 @"Computer keyboard",  // expectedLabel
                 nil                    // expectedDisplaName
  );
}

@end

NS_ASSUME_NONNULL_END
