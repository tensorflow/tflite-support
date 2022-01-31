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
import GMLImageHelpers
import XCTest

@testable import TFLObjectDetector

class TFLObjectDetectorTests: XCTestCase {

  static let bundle = Bundle(for: TFLObjectDetectorTests.self)
  static let modelPath = bundle.path(
    forResource: "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29",
    ofType: "tflite")!

  func verifyDetectionResult(_ detectionResult: TFLDetectionResult) {
    XCTAssertGreaterThan(detectionResult.detections.count, 0)

    self.verifyDetection(
      detectionResult.detections[0],
      expectedBoundingBox: CGRect(x: 54, y: 396, width: 393, height: 199),
      expectedFirstScore: 0.632812,
      expectedFirstLabel: "cat")

    self.verifyDetection(
      detectionResult.detections[1],
      expectedBoundingBox: CGRect(x: 602, y: 157, width: 394, height: 447),
      expectedFirstScore: 0.609375,
      expectedFirstLabel: "cat")

    self.verifyDetection(
      detectionResult.detections[2],
      expectedBoundingBox: CGRect(x: 260, y: 394, width: 179, height: 209),
      expectedFirstScore: 0.5625,
      expectedFirstLabel: "cat")

    self.verifyDetection(
      detectionResult.detections[3],
      expectedBoundingBox: CGRect(x: 387, y: 197, width: 281, height: 409),
      expectedFirstScore: 0.488281,
      expectedFirstLabel: "dog")
  }

  func verifyDetection(
    _ detection: TFLDetection, expectedBoundingBox: CGRect,
    expectedFirstScore: Float,
    expectedFirstLabel: String
  ) {
    XCTAssertGreaterThan(detection.categories.count, 0)
    XCTAssertEqual(
      detection.boundingBox.origin.x,
      expectedBoundingBox.origin.x)
    XCTAssertEqual(
      detection.boundingBox.origin.y,
      expectedBoundingBox.origin.y)
    XCTAssertEqual(
      detection.boundingBox.size.width,
      expectedBoundingBox.size.width)
    XCTAssertEqual(
      detection.boundingBox.size.height,
      expectedBoundingBox.size.height)
    XCTAssertEqual(
      detection.categories[0].label,
      expectedFirstLabel)
    XCTAssertEqualWithAccuracy(
      detection.categories[0].score,
      expectedFirstScore, accuracy: 0.001)
  }

  func testSuccessfullInferenceOnMLImageWithUIImage() throws {

    let modelPath = try XCTUnwrap(TFLObjectDetectorTests.modelPath)

    let objectDetectorOptions = TFLObjectDetectorOptions(modelPath: modelPath)
    XCTAssertNotNil(objectDetectorOptions)

    let objectDetector =
      try TFLObjectDetector.objectDetector(options: objectDetectorOptions!)

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "cats_and_dogs",
        type: "jpg"))
    let detectionResults: TFLDetectionResult =
      try objectDetector.detect(gmlImage: gmlImage)

    self.verifyDetectionResult(detectionResults)
  }

  func testModelOptionsWithMaxResults() throws {

    let modelPath = try XCTUnwrap(TFLObjectDetectorTests.modelPath)

    let objectDetectorOptions = TFLObjectDetectorOptions(modelPath: modelPath)
    XCTAssertNotNil(objectDetectorOptions)

    let maxResults = 3
    objectDetectorOptions!.classificationOptions.maxResults = maxResults

    let objectDetector =
      try TFLObjectDetector.objectDetector(options: objectDetectorOptions!)

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "cats_and_dogs",
        type: "jpg"))
    let detectionResult: TFLDetectionResult = try objectDetector.detect(
      gmlImage: gmlImage)

    XCTAssertLessThanOrEqual(detectionResult.detections.count, maxResults)

    self.verifyDetection(
      detectionResult.detections[0],
      expectedBoundingBox: CGRect(x: 54, y: 396, width: 393, height: 199),
      expectedFirstScore: 0.632812,
      expectedFirstLabel: "cat")

    self.verifyDetection(
      detectionResult.detections[1],
      expectedBoundingBox: CGRect(x: 602, y: 157, width: 394, height: 447),
      expectedFirstScore: 0.609375,
      expectedFirstLabel: "cat")

    self.verifyDetection(
      detectionResult.detections[2],
      expectedBoundingBox: CGRect(x: 260, y: 394, width: 179, height: 209),
      expectedFirstScore: 0.5625,
      expectedFirstLabel: "cat")
  }
}
