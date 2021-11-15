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
@testable import TFLImageClassifier

class TFLImageClassifierTests: XCTestCase {
  
  static let bundle = Bundle(for: TFLImageClassifierTests.self)
  static let modelPath = bundle.path(
    forResource: "mobilenet_v2_1.0_224",
    ofType: "tflite")!
  
  override func setUpWithError() throws {
      // Put setup code here. This method is called before the invocation of each test method in the class.
    }

  override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

  func testSuccessfullInferenceOnMLImageWithUIImage() throws {
      
    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)

    let imageClassifierOptions = TFLImageClassifierOptions(modelPath: modelPath)
    XCTAssertNotNil(imageClassifierOptions)
    
    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions!)
      
    let imagePath = try XCTUnwrap(TFLImageClassifierTests.bundle.path(forResource: "burger_crop", ofType: "jpg"))
    let image = UIImage(contentsOfFile: imagePath)
    let imageForInference = try XCTUnwrap(image)
    let gmlImage =  try XCTUnwrap(MLImage(image: imageForInference))
      
    let classificationResults: TFLClassificationResult = try imageClassifier.classify(gmlImage:gmlImage)
      
    XCTAssertNotNil(classificationResults)
    XCTAssert(classificationResults.classifications.count == 1
                && classificationResults.classifications[0].categories.count > 0
                && classificationResults.classifications[0].categories[0].label == "cheeseburger")

    }
  
  func testModelOptionsWithMaxResults() throws {
    
    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)
    
    let imageClassifierOptions = TFLImageClassifierOptions(modelPath: modelPath);
    XCTAssertNotNil(imageClassifierOptions)
    
    let maxResults = 3
    imageClassifierOptions!.classificationOptions.maxResults = maxResults
    
    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions!)
    
    let imagePath = try XCTUnwrap(TFLImageClassifierTests.bundle.path(forResource: "burger_crop", ofType: "jpg"))
    let image = UIImage(contentsOfFile: imagePath)
    let imageForInference = try XCTUnwrap(image)
    let gmlImage =  try XCTUnwrap(MLImage(image: imageForInference))
    
    let classificationResults: TFLClassificationResult = try imageClassifier.classify(gmlImage:gmlImage)
    
    XCTAssertNotNil(classificationResults)
    XCTAssert(classificationResults.classifications.count == 1
                && classificationResults.classifications[0].categories.count > 0
                && classificationResults.classifications[0].categories[0].label == "cheeseburger")
    
  }
  
  func testInferenceWithBoundingBox() throws {
    
    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)
    
    let imageClassifierOptions = TFLImageClassifierOptions(modelPath: modelPath)
    XCTAssertNotNil(imageClassifierOptions)

    let maxResults = 3
    imageClassifierOptions!.classificationOptions.maxResults = maxResults
    
    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions!)
    
    let imagePath = try XCTUnwrap(TFLImageClassifierTests.bundle.path(forResource: "burger_crop", ofType: "jpg"))
    let image = UIImage(contentsOfFile: imagePath)
    let imageForInference = try XCTUnwrap(image)
    
    let gmlImage =  try XCTUnwrap(MLImage(image: imageForInference))
    
    let roi = CGRect(x: 20, y: 20, width: 200, height: 200)
    let classificationResults =
      try imageClassifier.classify(gmlImage: gmlImage, regionOfInterest: roi)
    
    XCTAssertNotNil(classificationResults)
    XCTAssert(classificationResults.classifications.count == 1
                && classificationResults.classifications[0].categories.count > 0
                && classificationResults.classifications[0].categories[0].label == "cheeseburger")
    
  }
  
  func testInferenceWithRGBAImage() throws {
    
    let modelPath = try XCTUnwrap(TFLImageClassifierTests.modelPath)
  
    let imageClassifierOptions = TFLImageClassifierOptions(modelPath: modelPath)
    XCTAssertNotNil(imageClassifierOptions)
    
    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions!)
    
    let imagePath = try XCTUnwrap(TFLImageClassifierTests.bundle.path(forResource: "sparrow", ofType: "png"))
    let image = UIImage(contentsOfFile: imagePath)
    let imageForInference = try XCTUnwrap(image)
    let gmlImage =  try XCTUnwrap(MLImage(image: imageForInference))
    
    let classificationResults =
      try imageClassifier.classify(gmlImage: gmlImage)
    
    XCTAssertNotNil(classificationResults)
    XCTAssert(classificationResults.classifications.count == 1
                && classificationResults.classifications[0].categories.count > 0
                && classificationResults.classifications[0].categories[0].label == "junco")
    
  }

}
