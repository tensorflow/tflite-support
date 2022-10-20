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
import XCTest
import GMLImageUtils

@testable import TFLImageSearcher

class ImageSearcherTests: XCTestCase {

  static let bundle = Bundle(for: ImageSearcherTests.self)

  let kSearcherModelName = "mobilenet_v3_small_100_224_searcher"
  let kEmbedderModelName = "mobilenet_v3_small_100_224_embedder"
  let kMobileNetIndexName = "searcher_index"

  func validateError(
    _ error: Error,
    expectedLocalizedDescription: String
  ) {

    XCTAssertEqual(
      error.localizedDescription,
      expectedLocalizedDescription)
  }

  func validateNearestNeighbor(
    _ nearestNeighbor: NearestNeighbor,
    expectedMetadata: String,
    expectedDistance: Double
  ) {
    XCTAssertEqual(
      nearestNeighbor.metadata,
      expectedMetadata)
    XCTAssertEqual(
      nearestNeighbor.distance,
      expectedDistance,
      accuracy: 1e-6)
  }

  func validateSearchResultCount(
    _ searchResult: SearchResult,
    expectedNearestNeighborsCount: Int
  ) {
    XCTAssertEqual(
      searchResult.nearestNeighbors.count,
      expectedNearestNeighborsCount)
  }

  func filePath(
    name: String,
    fileExtension: String
  ) throws -> String? {
        
    return try XCTUnwrap(ImageSearcherTests.bundle.path(
        forResource: name,
        ofType: fileExtension))
  }

  func createImageSearcherOptions(
    modelName: String
  ) throws -> ImageSearcherOptions? {

    let modelPath = try XCTUnwrap(filePath(name: modelName,
        fileExtension: "tflite"))
    return ImageSearcherOptions(modelPath: modelPath)
  }

  func createImageSearcher(
    modelName: String,
    indexFileName: String? = nil
  ) throws -> ImageSearcher? {
    let options = try XCTUnwrap(
      self.createImageSearcherOptions(
        modelName: "mobilenet_v3_small_100_224_searcher"))
    

    if let _indexFileName = indexFileName {
      let indexFilePath = try XCTUnwrap(filePath(name: _indexFileName,
          fileExtension: "ldb"))
      options.searchOptions.indexFile.filePath = indexFilePath
    }

    let imageSearcher = try XCTUnwrap(
      ImageSearcher.searcher(
        options: options))

    return imageSearcher
  }
 

  func validateSearchResult(
    _ searchResult: SearchResult
  ) {
    self.validateSearchResultCount(
      searchResult,
      expectedNearestNeighborsCount: 5)
    
    self.validateNearestNeighbor(
      searchResult.nearestNeighbors[0],
      expectedMetadata: "burger",
      expectedDistance: 198.456329)
    self.validateNearestNeighbor(
      searchResult.nearestNeighbors[1],
      expectedMetadata: "car",
      expectedDistance: 226.022186)
    
    self.validateNearestNeighbor(
      searchResult.nearestNeighbors[2],
      expectedMetadata: "bird",
      expectedDistance: 227.297668)
    self.validateNearestNeighbor(
      searchResult.nearestNeighbors[3],
      expectedMetadata: "dog",
      expectedDistance: 229.133789)
   self.validateNearestNeighbor(searchResult.nearestNeighbors[4],
      expectedMetadata: "cat", 
      expectedDistance: 229.718948)
  }

  func testSearchWithSearcherModelSucceeds() throws {
    let imageSearcher = try XCTUnwrap(self.createImageSearcher(
      modelName: kSearcherModelName))
   
    let mlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "burger",
        type: "jpg"))

    let searchResult = try XCTUnwrap(
      imageSearcher.search(
        mlImage: mlImage))
    self.validateSearchResult(searchResult)
  }

  func testSearchWithEmbedderModelAndIndexFileSucceeds() throws {
    let imageSearcher = try XCTUnwrap(self.createImageSearcher(
      modelName: kEmbedderModelName,
      indexFileName: kMobileNetIndexName))
   
    let mlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "burger",
        type: "jpg"))

    let searchResult = try XCTUnwrap(
      imageSearcher.search(
        mlImage: mlImage))
    self.validateSearchResult(searchResult)
  }

  func testCreateImageSearcherWithNoModelPathFails() throws {
    let options = ImageSearcherOptions()
    do {
      let imageSearcher = try ImageSearcher.searcher(
        options: options)
      XCTAssertNil(imageSearcher)
    } catch {
      self.validateError(
        error,
        expectedLocalizedDescription:
          "INVALID_ARGUMENT: Missing mandatory `model_file` field in `base_options`")
    }
  }
}
