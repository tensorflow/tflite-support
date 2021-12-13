//
//  TFLDetectionResult.h
//  ObjectDetection
//
//  Created by Prianka Kariat on 13/12/21.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "tensorflow_lite_support/ios/task/processor/sources/TFLClassificationOptions.h"

NS_ASSUME_NONNULL_BEGIN

/** Encapsulates list of predicted classes (aka labels) and bounding box for a detected object. */
@interface TFLDetection : NSObject

/**
 * The index of the image classifier head these classes refer to. This is useful for multi-head
 * models.
 */
@property(nonatomic, assign) CGRect boundingBox;

/** The array of predicted classes, usually sorted by descending scores (e.g.from high to low
 * probability). */
@property(nonatomic, copy) NSArray<TFLCategory *> *categories;

@end

/** Encapsulates results of any object detection task. */
@interface TFLDetectionResult : NSObject

@property(nonatomic, copy) NSArray<TFLDetection *> *detections;

@end

NS_ASSUME_NONNULL_END
