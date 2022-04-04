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
import GMLImageUtils
import XCTest

@testable import TFLImageClassifier

class TFLImageClassifierTests: XCTestCase {

  static let bundle = Bundle(for: TFLImageClassifierTests.self)
  static let modelPath = bundle.path(
    forResource: "mobilenet_v2_1.0_224",
    ofType: "tflite")

  func verifyCategory(
    _ category: TFLCategory,
    expectedIndex: NSInteger,
    expectedScore: Float,
    expectedLabel: String,
    expectedDisplayName: String?
  ) {
    XCTAssertEqual(
      category.index,
      expectedIndex)
    XCTAssertEqual(
      category.score, 
      expectedScore, 
      accuracy: 1e-6); 

    XCTAssertEqual(
      category.label,
      expectedLabel)
    
    XCTAssertEqual(
      category.displayName,
      expectedDisplayName)
  }

  func verifyClassifications(
    _ classifications: TFLClassifications,
    expectedHeadIndex: NSInteger,
    expectedCategoryCount: NSInteger
  ) {
    XCTAssertEqual(
      classifications.headIndex,
      expectedHeadIndex)
    XCTAssertEqual(
      classifications.categories.count,
      expectedCategoryCount)
  }

  func verifyClassificationResult(
    _ classificationResult: TFLClassificationResult,
    expectedClassificationsCount: NSInteger
  ) {
    XCTAssertEqual(
      classificationResult.classifications.count,
      expectedClassificationsCount)
  }

  func testSuccessfullInferenceOnMLImageWithUIImage() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))

    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions)

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "burger",
        type: "jpg"))

    let classificationResult: TFLClassificationResult =
      try XCTUnwrap(imageClassifier.classify(gmlImage: gmlImage))
    
    let expectedClassificationsCount = 1
    self.verifyClassificationResult(classificationResult, 
                                    expectedClassificationsCount: 1)

    let expectedCategoryCount = 1001
    let expectedHeadIndex = 0
    self.verifyClassifications(classificationResult.classifications[0], 
                               expectedHeadIndex: expectedHeadIndex, 
                               expectedCategoryCount: expectedCategoryCount)
    
    // TODO: match the score as image_classifier_test.cc
    self.verifyCategory(classificationResult.classifications[0].categories[0], 
                        expectedIndex: 934, 
                        expectedScore: 0.748976,
                        expectedLabel: "cheeseburger", 
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[1], 
                        expectedIndex: 925, 
                        expectedScore: 0.024646, 
                        expectedLabel: "guacamole",
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[2], 
                        expectedIndex: 932, 
                        expectedScore: 0.022505, 
                        expectedLabel: "bagel",
                        expectedDisplayName: nil);
  }

  func testModelOptionsWithMaxResults() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))
    XCTAssertNotNil(imageClassifierOptions)

    let maxResults = 3
    imageClassifierOptions.classificationOptions.maxResults = maxResults

    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions)

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "burger",
        type: "jpg"))

    let classificationResult: TFLClassificationResult = try imageClassifier.classify(
      gmlImage: gmlImage)

    let expectedClassificationsCount = 1
    self.verifyClassificationResult(classificationResult, 
                                    expectedClassificationsCount: 1)

    let expectedHeadIndex = 0
    self.verifyClassifications(classificationResult.classifications[0], 
                               expectedHeadIndex: expectedHeadIndex, 
                               expectedCategoryCount: maxResults)
    
    // TODO: match the score as image_classifier_test.cc
    self.verifyCategory(classificationResult.classifications[0].categories[0], 
                        expectedIndex: 934, 
                        expectedScore: 0.748976,
                        expectedLabel: "cheeseburger", 
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[1], 
                        expectedIndex: 925, 
                        expectedScore: 0.024646, 
                        expectedLabel: "guacamole",
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[2], 
                        expectedIndex: 932, 
                        expectedScore: 0.022505, 
                        expectedLabel: "bagel",
                        expectedDisplayName: nil);
  }

  func testInferenceWithBoundingBox() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))

    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions)

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "multi_objects",
        type: "jpg"))

    let roi = CGRect(x: 406, y: 110, width: 148, height: 153)
    let classificationResults =
      try imageClassifier.classify(gmlImage: gmlImage, regionOfInterest: roi)

    let expectedClassificationsCount = 1
    self.verifyClassificationResult(classificationResult, 
                                    expectedClassificationsCount: 1)

    let expectedHeadIndex = 0
    self.verifyClassifications(classificationResult.classifications[0], 
                               expectedHeadIndex: expectedHeadIndex, 
                               expectedCategoryCount: maxResults)
    
    // TODO: match the score as image_classifier_test.cc
    self.verifyCategory(classificationResult.classifications[0].categories[0], 
                        expectedIndex: 934, 
                        expectedScore: 0.748976,
                        expectedLabel: "cheeseburger", 
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[1], 
                        expectedIndex: 925, 
                        expectedScore: 0.024646, 
                        expectedLabel: "guacamole",
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[2], 
                        expectedIndex: 932, 
                        expectedScore: 0.022505, 
                        expectedLabel: "bagel",
                        expectedDisplayName: nil);
  }

    // TODO: match the label and score as image_classifier_test.cc
    // let category = classificationResults.classifications[0].categories[0]
    // XCTAssertEqual(category.label, "soccer ball")
    // XCTAssertEqual(category.score, 0.256512, accuracy:0.001);
  }

  func testInferenceWithRGBAImage() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))

    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions)

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "sparrow",
        type: "png"))

    let classificationResults =
      try imageClassifier.classify(gmlImage: gmlImage)

    XCTAssertNotNil(classificationResults)
    XCTAssertEqual(classificationResults.classifications.count, 1)
    XCTAssertGreaterThan(classificationResults.classifications[0].categories.count, 0)

    let category = classificationResults.classifications[0].categories[0]
    XCTAssertEqual(category.label, "junco")
    XCTAssertEqual(category.score, 0.253016, accuracy: 0.001)
  }
}
