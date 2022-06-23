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
import AVAudioPCMBufferUtils
import XCTest

@testable import TFLAudioClassifier

class AudioClassifierTests: XCTestCase {

  static let bundle = Bundle(for: AudioClassifierTests.self)
  static let modelPath = bundle.path(
    forResource: "yamnet_audio_classifier_with_metadata",
    ofType: "tflite")
  
  func bufferFromFile(name: String, fileExtension:String, audioFormat:AudioFormat) -> AVAudioPCMBuffer? {
    guard let filePath = AudioClassifierTests.bundle.path(
      forResource: name,
      ofType: fileExtension) else {
        return nil;
      }

    return AVAudioPCMBuffer.loadPCMBufferFromFile(withPath:filePath, audioFormat:audioFormat)
  }
  func testInferenceWithFloatBufferSucceeds() throws {

    let modelPath = try XCTUnwrap(AudioClassifierTests.modelPath)

    let options = AudioClassifierOptions(modelPath: modelPath)

    let audioClassifier =
      try AudioClassifier.classifier(options: options)

  }

}
