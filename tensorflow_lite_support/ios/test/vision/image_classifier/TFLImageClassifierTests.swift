//
//  TFLImageClassifierTests.swift
//  TFLTaskImageClassifierFrameworkTests
//
//  Created by Prianka Kariat on 31/08/21.
//

import XCTest
@testable import TFLTaskImageClassifierFramework


class TFLImageClassifierSwiftTests: XCTestCase {
  
    static let bundle = Bundle(for: TFLImageClassifierTest.self)
    static let modelPath =
      Bundle.main.path(forResource: "mobilenet_v2_1.0_224",
                       ofType: "tflite")

    override func setUpWithError() throws {
      // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testSuccessfullInferenceOnMLImageWithUIImage() throws {
      
      let modelPath = try XCTUnwrap(TFLImageClassifierSwiftTests.modelPath)

      let imageClassifier = try TFLImageClassifier.imageClassifier(modelPath: modelPath)
      
      let image = UIImage(named: "burger_crop" )
      let imageForInference = try XCTUnwrap(image)
      let gmlImage =  try XCTUnwrap(MLImage(image: imageForInference))
      
      let classificationResults: TFLClassificationResult = try imageClassifier.classify(gmlImage:gmlImage)
      
      XCTAssertNotNil(classificationResults)
      XCTAssert(classificationResults.classifications.count == 1
                  && classificationResults.classifications[0].categories.count > 0
                  && classificationResults.classifications[0].categories[0].label == "cheeseburger")

    }
  
  func testModelOptionsWithMaxResults() throws {
    
    let modelPath = try XCTUnwrap(TFLImageClassifierSwiftTests.modelPath)
    
    let imageClassifierOptions = TFLImageClassifierOptions(modelPath: modelPath);
    XCTAssertNotNil(imageClassifierOptions)
    
    let maxResults = 3
    imageClassifierOptions!.classificationOptions.maxResults = maxResults
    
    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions!)
    
    let image = UIImage(named: "burger_crop" )
    let imageForInference = try XCTUnwrap(image)
    let gmlImage =  try XCTUnwrap(MLImage(image: imageForInference))
    
    let classificationResults: TFLClassificationResult = try imageClassifier.classify(gmlImage:gmlImage)
    
    XCTAssertNotNil(classificationResults)
    XCTAssert(classificationResults.classifications.count == 1
                && classificationResults.classifications[0].categories.count > 0
                && classificationResults.classifications[0].categories[0].label == "cheeseburger")
    
  }
  
  func testInferenceWithBoundingBox() throws {
    
    let modelPath = try XCTUnwrap(TFLImageClassifierSwiftTests.modelPath)
    
    let imageClassifierOptions = TFLImageClassifierOptions(modelPath: modelPath)
    XCTAssertNotNil(imageClassifierOptions)

    let maxResults = 3
    imageClassifierOptions!.classificationOptions.maxResults = maxResults
    
    let imageClassifier =
      try TFLImageClassifier.imageClassifier(options: imageClassifierOptions!)
    
    let image = UIImage(named: "burger_crop" )
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
    
    let modelPath = try XCTUnwrap(TFLImageClassifierSwiftTests.modelPath)
  
    let imageClassifier =
      try TFLImageClassifier.imageClassifier(modelPath: modelPath)
    
    let image = UIImage(named: "sparrow" )
    let imageForInference = try XCTUnwrap(image)
    let gmlImage =  try XCTUnwrap(MLImage(image: imageForInference))
    
    let classificationResults =
      try imageClassifier.classify(gmlImage: gmlImage)
    
    XCTAssertNotNil(classificationResults)
    XCTAssert(classificationResults.classifications.count == 1
                && classificationResults.classifications[0].categories.count > 0
                && classificationResults.classifications[0].categories[0].label == "brambling")
    
  }

}
