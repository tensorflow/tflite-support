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
  
  func verifyError(
    _ error: Error,
    expectedLocalizedDescription: String
  ) {
    XCTAssertEqual(
      error.localizedDescription,
      expectedLocalizedDescription)
  }

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

  func testInferenceOnMLImageWithUIImage() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))

    let imageClassifier =
      try XCTUnwrap(TFLImageClassifier.imageClassifier(options: imageClassifierOptions))

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "burger",
        type: "jpg"))

    let classificationResult: TFLClassificationResult =
      try XCTUnwrap(imageClassifier.classify(gmlImage: gmlImage))
    
    let expectedClassificationsCount = 1
    self.verifyClassificationResult(classificationResult, 
                                    expectedClassificationsCount: expectedClassificationsCount)

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

  func testErrorForSimultaneousLabelAllowListAndDenyList() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))
    imageClassifierOptions.classificationOptions.labelAllowList = ["cheeseburger"];
    imageClassifierOptions.classificationOptions.labelDenyList = ["cheeseburger"];

    var imageClassifier: TFLImageClassifier?
    do {
      let imageClassifier =
        try TFLImageClassifier.imageClassifier(options: imageClassifierOptions)
      XCTAssertNil(imageClassifier)
    }
    catch  {
      let expectedLocalizedDescription =
        "INVALID_ARGUMENT: `class_name_whitelist` and `class_name_blacklist` are mutually exclusive options"
      self.verifyError(error,
                       expectedLocalizedDescription: expectedLocalizedDescription)
    }
  }
    
  func testModelOptionsWithMaxResults() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))
    XCTAssertNotNil(imageClassifierOptions)

    let maxResults = 3
    imageClassifierOptions.classificationOptions.maxResults = maxResults

    let imageClassifier =
      try XCTUnwrap(TFLImageClassifier.imageClassifier(options: imageClassifierOptions))

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "burger",
        type: "jpg"))

    let classificationResult = 
      try XCTUnwrap(imageClassifier.classify(gmlImage: gmlImage))

    let expectedClassificationsCount = 1
    self.verifyClassificationResult(classificationResult, 
                                    expectedClassificationsCount: expectedClassificationsCount)

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

    func testErrorForOptionsWithInvalidMaxResults() throws {
    
    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)
    
    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))
    XCTAssertNotNil(imageClassifierOptions)
    
    let maxResults = 0
    imageClassifierOptions.classificationOptions.maxResults = maxResults
    
    do {
      let imageClassifier =
        try TFLImageClassifier.imageClassifier(options: imageClassifierOptions)
        XCTAssertNil(imageClassifier)
    }
    catch {
      let expectedLocalizedDescription =
        "INVALID_ARGUMENT: Invalid `max_results` option: value must be != 0"
      self.verifyError(error,
                       expectedLocalizedDescription: expectedLocalizedDescription)
    }
  }

  func testInferenceWithBoundingBox() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))

    let maxResults = 3
    imageClassifierOptions.classificationOptions.maxResults = maxResults

    let imageClassifier =
      try XCTUnwrap(TFLImageClassifier.imageClassifier(options: imageClassifierOptions))

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "multi_objects",
        type: "jpg"))

    let roi = CGRect(x: 406, y: 110, width: 148, height: 153)
    let classificationResult =
      try XCTUnwrap(imageClassifier.classify(gmlImage: gmlImage, regionOfInterest: roi))

    let expectedClassificationsCount = 1
    self.verifyClassificationResult(classificationResult, 
                                    expectedClassificationsCount: expectedClassificationsCount)

    let expectedHeadIndex = 0
    self.verifyClassifications(classificationResult.classifications[0], 
                               expectedHeadIndex: expectedHeadIndex, 
                               expectedCategoryCount: maxResults)
    
    // TODO: match the score as image_classifier_test.cc
    self.verifyCategory(classificationResult.classifications[0].categories[0], 
                        expectedIndex: 806, 
                        expectedScore: 0.997143,
                        expectedLabel: "soccer ball", 
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[1], 
                        expectedIndex: 891, 
                        expectedScore: 0.000380, 
                        expectedLabel: "volleyball",
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[2], 
                        expectedIndex: 685, 
                        expectedScore: 0.000198, 
                        expectedLabel: "ocarina",
                        expectedDisplayName: nil);
  }

  func testInferenceWithRGBAImage() throws {

    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = try XCTUnwrap(TFLImageClassifierOptions(modelPath: modelPath))

    let imageClassifier =
      try XCTUnwrap(TFLImageClassifier.imageClassifier(options: imageClassifierOptions))

    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "burger",
        type: "png"))

    let classificationResult =
      try XCTUnwrap(imageClassifier.classify(gmlImage: gmlImage))

   self.verifyCategory(classificationResult.classifications[0].categories[0], 
                        expectedIndex: 934, 
                        expectedScore: 0.738065,
                        expectedLabel: "cheeseburger", 
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[1], 
                        expectedIndex: 925, 
                        expectedScore: 0.027371, 
                        expectedLabel: "guacamole",
                        expectedDisplayName: nil);
    self.verifyCategory(classificationResult.classifications[0].categories[2], 
                        expectedIndex: 932, 
                        expectedScore: 0.026174, 
                        expectedLabel: "bagel",
                        expectedDisplayName: nil);
    }

}
