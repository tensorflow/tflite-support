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
import CoreGraphics
import XCTest

@testable import TFLImageSegmenter

class TFLImageSegmenterTests: XCTestCase {

  // The maximum fraction of pixels in the candidate mask that can have a
  // different class than the golden mask for the test to pass.
  let kGoldenMaskTolerance: Float = 1e-2;
  // Magnification factor used when creating the golden category masks to make
  // them more human-friendly. Each pixel in the golden masks has its value
  // multiplied by this factor, i.e. a value of 10 means class index 1, a value of
  // 20 means class index 2, etc.
  let kGoldenMaskMagnificationFactor: UInt = 10;
  
  static let bundle = Bundle(for: TFLImageSegmenterTests.self)
  static let modelPath = bundle.path(
    forResource: "deeplabv3",
    ofType: "tflite")!

  func testSuccessfullInferenceOnMLImageWithUIImage() throws {
    
    let modelPath = try XCTUnwrap(TFLImageSegmenterTests.modelPath)
    
    let imageSegmenterOptions = TFLImageSegmenterOptions(modelPath: modelPath)
    XCTAssertNotNil(imageSegmenterOptions)
    
    
    let imageSegmenter =
    try TFLImageSegmenter.imageSegmenter(options: imageSegmenterOptions!)
    
    let gmlImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "segmentation_input_rotation0",
        type: "jpg"))
    
    let segmentationResults: TFLSegmentationResult =
    try imageSegmenter.segment(gmlImage: gmlImage)
    
    let goldenImage = try XCTUnwrap(
      MLImage.imageFromBundle(
        class: type(of: self),
        filename: "segmentation_golden_rotation0",
        type: "png"))

    let cvPixelBuffer: CVPixelBuffer = goldenImage.grayScalePixelBufferFromUIImage().takeRetainedValue();

    CVPixelBufferLockBaseAddress(cvPixelBuffer, CVPixelBufferLockFlags.readOnly);
    
    let baseAddress =  try XCTUnwrap(CVPixelBufferGetBaseAddress(cvPixelBuffer))
    let pixelBufferBaseAddress = baseAddress.assumingMemoryBound(to: UInt8.self)

    // Category Mask is TFLCategoryMask
    let categoryMask = try XCTUnwrap(segmentationResults.segmentations[0].categoryMask)

    XCTAssertEqual(CVPixelBufferGetWidth(cvPixelBuffer), categoryMask.width)
    XCTAssertEqual(CVPixelBufferGetHeight(cvPixelBuffer), categoryMask.height)
   
    let num_pixels = CVPixelBufferGetWidth(cvPixelBuffer) * CVPixelBufferGetHeight(cvPixelBuffer)

    var inconsistentPixels = 0
    
    for i in 0..<num_pixels {
      
      if categoryMask.mask[i] *
          kGoldenMaskMagnificationFactor !=
          pixelBufferBaseAddress[i] {
          inconsistentPixels += 1
      }
    }
    CVPixelBufferUnlockBaseAddress(cvPixelBuffer, CVPixelBufferLockFlags.readOnly)

    XCTAssertLessThan(Float(inconsistentPixels)/Float(num_pixels), kGoldenMaskTolerance)
  }

}
