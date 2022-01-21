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
import XCTest

@testable import TFLObjectDetector
import GMLImageHelpers
import TFLTestUtil

class TFLObjectDetectorTests: XCTestCase {

  static let bundle = Bundle(for: TFLObjectDetectorTests.self)
  static let modelPath = bundle.path(
    forResource: "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29",
    ofType: "tflite")!

  func testSuccessfullInferenceOnMLImageWithUIImage() throws {

    let modelPath = try XCTUnwrap(TFLObjectDetectorTests.modelPath)

    let objectDetectorOptions = TFLObjectDetectorOptions(modelPath: modelPath)
    XCTAssertNotNil(objectDetectorOptions)

    let objectDetector =
      try TFLObjectDetector.objectDetector(options: objectDetectorOptions!)
    
    
    let gmlImage = try XCTUnwrap(MLImage.imageFromBundle(class: type(of:self), 
                                             filename: "cats_and_dogs", 
                                                 type: "jpg" ))
    let detectionResults: TFLDetectionResult =
      try objectDetector.detect(gmlImage: gmlImage)

    TFLTestUtils.verify(detectionResult: detectionResults);
  }

  func testModelOptionsWithMaxResults() throws {

    let modelPath = try XCTUnwrap(TFLObjectDetectorTests.modelPath)

    let objectDetectorOptions = TFLObjectDetectorOptions(modelPath: modelPath)
    XCTAssertNotNil(objectDetectorOptions)

    let maxResults = 3
    objectDetectorOptions!.classificationOptions.maxResults = maxResults

    let objectDetector =
      try TFLObjectDetector.objectDetector(options: objectDetectorOptions!)

    let gmlImage = try XCTUnwrap(MLImage.imageFromBundle(class: type(of:self), 
                                             filename: "cats_and_dogs", 
                                                 type: "jpg" ))
    let detectionResult: TFLDetectionResult = try objectDetector.detect(
      gmlImage: gmlImage)

    XCTAssertLessThanOrEqual(detectionResult.detections.count, maxResults);

    TFLTestUtils.verify(detection: detectionResult.detections[0],
              expectedBoundingBox: CGRect(x:54, y:396, width:393, height:199), 
               expectedFirstScore: 0.632812,                                    
               expectedFirstLabel: "cat");                                      
  
    TFLTestUtils.verify(detection: detectionResult.detections[1],
              expectedBoundingBox: CGRect(x:602, y:157, width:394, height:447), 
               expectedFirstScore: 0.609375,                                    
               expectedFirstLabel: "cat");  

    TFLTestUtils.verify(detection: detectionResult.detections[2],
              expectedBoundingBox: CGRect(x:260, y:394, width:179, height:209), 
               expectedFirstScore: 0.5625,                                    
               expectedFirstLabel: "cat");  
  } 
}

  