//
//  TFLDetectionUtils.h
//  ObjectDetection
//
//  Created by Prianka Kariat on 13/12/21.
//

#import <Foundation/Foundation.h>
#import "tensorflow_lite_support/ios/task/processor/sources/TFLDetectionResult.h"

#include "tensorflow_lite_support/c/task/processor/detection_result.h"

NS_ASSUME_NONNULL_BEGIN

/** Helper utility for conversion between TFLite Task C Library Detection Results and iOS
 * Detection Results . */
@interface TFLDetectionUtils : NSObject

/**
 * Creates and retrurns a TFLDetectionResult from a TfLiteDetectionResult returned by
 * TFLite Task C Library Object Detection task.
 *
 * @param cDetectionResult Detection  results returned by TFLite Task C Library
 * Object Detection task.
 *
 * @return Detection Result of type TFLDetectionResult to be returned by inference methods
 * of the iOS TF Lite Task Object Detection task.
 */
+ (TFLDetectionResult *)detectionResultFromCDetectionResults:
(TfLiteDetectionResult *)cDetectionResult;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
