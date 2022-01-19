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
import TFLTestUtil

class TFLObjectDetectorTests: XCTestCase {

  
  static let bundle = Bundle(for: TFLObjectDetectorTests.self)
  static let modelPath = bundle.path(
    forResource: "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29",
    ofType: "tflite")!

  func testSuccessfullInferenceOnMLImageWithUIImage() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = TFLImageClassifierOptions(modelPath: modelPath)
    XCTAssertNotNil(imageClassifierOptions)

    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions!)

    let gmlImage = try gmlImage(withName: "cats_and_dogs", ofType: "jpg")
    let classificationResults: TFLClassificationResult =
      try imageClassifier.classify(gmlImage: gmlImage)

    XCTAssertNotNil(classificationResults)
    XCTAssertEqual(classificationResults.classifications.count, 1)
    XCTAssertGreaterThan(classificationResults.classifications[0].categories.count, 0)
    // TODO: match the score as image_classifier_test.cc
    let category = classificationResults.classifications[0].categories[0]
    XCTAssertEqual(category.label, "cheeseburger")
    XCTAssertEqual(category.score, 0.748976, accuracy: 0.001)
  }

  func testModelOptionsWithMaxResults() throws {

    let modelPath = try XCTUnwrap(TFLObjectDetectorTests.modelPath)

    let objectDetectorOptions = TFLObjectDetectorOptions(modelPath: modelPath)
    XCTAssertNotNil(objectDetectorOptions)

    let maxResults = 3
    objectDetectorOptions!.classificationOptions.maxResults = maxResults

    let objectDetector =
      try TFLObjectDetector.objectDetector(options: objectDetectorOptions!)

    let gmlImage = try gmlImage(withName: "cats_and_dogs", ofType: "jpg")

    let detectionResult: TFLDetectionResult = try objectDetector.detect(
      gmlImage: gmlImage)

    XCTAssertNotNil(detectionResult)
    XCTAssertEqual(detectionResult.detect.count, 1)
    XCTAssertLessThanOrEqual(classificationResults.classifications[0].categories.count, maxResults)

    // TODO: match the score as image_classifier_test.cc
    let category = classificationResults.classifications[0].categories[0]
    XCTAssertEqual(category.label, "cheeseburger")
    XCTAssertEqual(category.score, 0.748976, accuracy: 0.001)
  }

  func testInferenceWithBoundingBox() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = TFLImageClassifierOptions(modelPath: modelPath)
    XCTAssertNotNil(imageClassifierOptions)

    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions!)

    let gmlImage = try gmlImage(withName: "multi_objects", ofType: "jpg")

    let roi = CGRect(x: 406, y: 110, width: 148, height: 153)
    let classificationResults =
      try imageClassifier.classify(gmlImage: gmlImage, regionOfInterest: roi)

    XCTAssertNotNil(classificationResults)
    XCTAssertEqual(classificationResults.classifications.count, 1)
    XCTAssertGreaterThan(classificationResults.classifications[0].categories.count, 0)

    // TODO: match the label and score as image_classifier_test.cc
    // let category = classificationResults.classifications[0].categories[0]
    // XCTAssertEqual(category.label, "soccer ball")
    // XCTAssertEqual(category.score, 0.256512, accuracy:0.001);
  }

  func testInferenceWithRGBAImage() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = TFLImageClassifierOptions(modelPath: modelPath)
    XCTAssertNotNil(imageClassifierOptions)

    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions!)

    let gmlImage = try gmlImage(withName: "sparrow", ofType: "png")

    let classificationResults =
      try imageClassifier.classify(gmlImage: gmlImage)

    XCTAssertNotNil(classificationResults)
    XCTAssertEqual(classificationResults.classifications.count, 1)
    XCTAssertGreaterThan(classificationResults.classifications[0].categories.count, 0)

    let category = classificationResults.classifications[0].categories[0]
    XCTAssertEqual(category.label, "junco")
    XCTAssertEqual(category.score, 0.253016, accuracy: 0.001)
  }

  private func gmlImage(withName name: String, ofType type: String) throws -> MLImage {
    let imagePath =
      try XCTUnwrap(TFLImageClassifierTests.bundle.path(forResource: name, ofType: type))
    let image = UIImage(contentsOfFile: imagePath)
    let imageForInference = try XCTUnwrap(image)
    let gmlImage = try XCTUnwrap(MLImage(image: imageForInference))

    return gmlImage
  }
}
