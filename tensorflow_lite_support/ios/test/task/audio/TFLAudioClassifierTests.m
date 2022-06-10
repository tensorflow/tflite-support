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
#import <XCTest/XCTest.h>

#import "tensorflow_lite_support/ios/task/audio/sources/TFLAudioClassifier.h"
// #import "tensorflow_lite_support/ios/task/audio/utils/sources/GMLImage+Utils.h"

NS_ASSUME_NONNULL_BEGIN

@interface TFLAudioClassifierTests : XCTestCase
@property(nonatomic, nullable) NSString *modelPath;
@end

// This category of TFLAudioRecord is private to the current test file.
// This is needed in order to expose the method to load the audio record buffer
// without calling: -[TFLAudioRecord startRecordingWithError:].
// This is needed to avoid exposing this method which isn't useful to the consumers
// of the framework.
@interface TFLAudioRecord (Tests)
- (void)convertAndLoadBuffer:(AVAudioPCMBuffer *)buffer
         usingAudioConverter:(AVAudioConverter *)audioConverter;
@end


@implementation TFLAudioClassifierTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class.
  [super setUp];
  self.modelPath = [[NSBundle bundleForClass:self.class] pathForResource:@"yamnet_audio_classifier_with_metadata"
                                                                  ofType:@"tflite"];
  XCTAssertNotNil(self.modelPath);
}


- (AVAudioPCMBuffer *)audioEngineBufferFromFileWithName:(NSString *)name extension:(NSString *)extension {
  // Loading AVAudioPCMBuffer with an array is not currently suupported for iOS versions < 15.0.
  // Instead audio samples from a wav file are loaded and converted into the same format
  // of AVAudioEngine's input node to mock thhe input from the AVAudio Engine.
  NSString *filePath = [[NSBundle bundleForClass:self.class] pathForResource:name ofType:extension];

  return [AVAudioPCMBuffer loadPCMBufferFromFileWithPath:filePath
                                        processingFormat:self.audioEngineFormat];
}


- (void)testInferenceWithAudioRecordSucceeds {

  TFLAudioClassifier *audioClassifier =
      [[TFLAudioClassifier alloc] initWithModelPath:self.modelPath error:nil];
  XCTAssertNotNil(audioClassifier);

  TFLAudioRecord *audioRecord = [audioClassifier createAudioRecordWithError:nil];

  // Loading AVAudioPCMBuffer with an array is not currently supported for iOS versions < 15.0.
  // Instead audio samples from a wav file are loaded and converted into the same format
  // of AVAudioEngine's input node to mock thhe input from the AVAudio Engine.
  AVAudioPCMBuffer *audioEngineBuffer = [self audioEngineBufferFromFileWithName:@"speech"
                                                                      extension:@"wav"];
  XCTAssertNotNil(audioEngineBuffer);

  AVAudioFormat *recordingFormat =
      [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatFloat32
                                       sampleRate:audioRecord.audioFormat.sampleRate
                                         channels:(AVAudioChannelCount)audioRecord.audioFormat.channelCount
                                      interleaved:YES];

  AVAudioConverter *audioConverter = [[AVAudioConverter alloc] initFromFormat:self.audioEngineFormat
                                                                     toFormat:recordingFormat];
  // Convert and load the buffer of `TFLAudioRecord`.
  [audioRecord convertAndLoadBuffer:audioEngineBuffer usingAudioConverter:audioConverter];

  TFLAudioTensor *audioTensor = [audioClassifier createAudioRecordWithError:nil];
  [audioTensor loadAudioRecord:audioRecord withError:nil];

  TFLClassificationResult *result = [audioClassifier classifyWithAudioTensor:audioTensor error:nil];





  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"burger" ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResults = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                   error:nil];
  XCTAssertTrue(classificationResults.classifications.count > 0);
  XCTAssertTrue(classificationResults.classifications[0].categories.count > 0);

  TFLCategory *category = classificationResults.classifications[0].categories[0];
  XCTAssertTrue([category.label isEqual:@"cheeseburger"]);
  // TODO: match the score as image_classifier_test.cc
  XCTAssertEqualWithAccuracy(category.score, 0.748976, 0.001);
}

- (void)testModelOptionsWithMaxResults {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];
  int maxResults = 3;
  imageClassifierOptions.classificationOptions.maxResults = maxResults;

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"burger" ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResults = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                   error:nil];
  XCTAssertTrue(classificationResults.classifications.count > 0);
  XCTAssertLessThanOrEqual(classificationResults.classifications[0].categories.count, maxResults);

  TFLCategory *category = classificationResults.classifications[0].categories[0];
  XCTAssertTrue([category.label isEqual:@"cheeseburger"]);
  // TODO: match the score as image_classifier_test.cc
  XCTAssertEqualWithAccuracy(category.score, 0.748976, 0.001);
}

- (void)testInferenceWithBoundingBox {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];
  int maxResults = 3;
  imageClassifierOptions.classificationOptions.maxResults = maxResults;

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"multi_objects" ofType:@"jpg"];
  XCTAssertNotNil(gmlImage);

  CGRect roi = CGRectMake(406, 110, 148, 153);
  TFLClassificationResult *classificationResults = [imageClassifier classifyWithGMLImage:gmlImage
                                                                        regionOfInterest:roi
                                                                                   error:nil];
  XCTAssertTrue(classificationResults.classifications.count > 0);
  XCTAssertTrue(classificationResults.classifications[0].categories.count > 0);

  TFLCategory *category = classificationResults.classifications[0].categories[0];
  // TODO: match the label and score as image_classifier_test.cc
  // XCTAssertTrue([category.label isEqual:@"soccer ball"]);
  // XCTAssertEqualWithAccuracy(category.score, 0.256512, 0.001);
}

- (void)testInferenceWithRGBAImage {
  TFLImageClassifierOptions *imageClassifierOptions =
      [[TFLImageClassifierOptions alloc] initWithModelPath:self.modelPath];

  TFLImageClassifier *imageClassifier =
      [TFLImageClassifier imageClassifierWithOptions:imageClassifierOptions error:nil];
  XCTAssertNotNil(imageClassifier);

  GMLImage *gmlImage =
      [GMLImage imageFromBundleWithClass:self.class fileName:@"sparrow" ofType:@"png"];
  XCTAssertNotNil(gmlImage);

  TFLClassificationResult *classificationResults = [imageClassifier classifyWithGMLImage:gmlImage
                                                                                   error:nil];
  XCTAssertTrue(classificationResults.classifications.count > 0);
  XCTAssertTrue(classificationResults.classifications[0].categories.count > 0);

  TFLCategory *category = classificationResults.classifications[0].categories[0];
  XCTAssertTrue([category.label isEqual:@"junco"]);
  // TODO: inspect if score is correct. Better to test againest "burger", because we know the
  // expected result for "burger.jpg".
  XCTAssertEqualWithAccuracy(category.score, 0.253016, 0.001);
}

@end

NS_ASSUME_NONNULL_END
