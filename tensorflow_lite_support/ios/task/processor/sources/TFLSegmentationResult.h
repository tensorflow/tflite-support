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
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/** Holds a label associated with an RGB color, for display purposes. */
@interface TFLColoredLabel : NSObject

/** The RGB color components for the label, in the [0, 255] range. */
@property(nonatomic, assign) NSUInteger r;
@property(nonatomic, assign) NSUInteger g;
@property(nonatomic, assign) NSUInteger b;

/** The class name, as provided in the label map packed in the TFLite Model
 * Metadata.
 */
@property(nonatomic, copy) NSString *label;

/** The display name, as provided in the label map (if available) packed in
 * the TFLite Model Metadata. See `display_names_locale` field in
 * ImageSegmenterOptions.
 */
@property(nonatomic, copy) NSString *displayName;

@end

/** Encapsulates a resulting segmentation mask and associated metadata. */
@interface TFLSegmentation : NSObject

/**
 * The width of the mask. This is an intrinsic parameter of the model being
 * used, and does not depend on the input image dimensions.
 */
@property(nonatomic, assign) NSInteger width;

/**
 *  The height of the mask. This is an intrinsic parameter of the model being
 * used, and does not depend on the input image dimensions.
 */
@property(nonatomic, assign) NSInteger height;

/**
 *  One confidence masks of size `width` x `height` for each of the supported
 * classes. The value of each pixel in these masks represents the confidence
 * score for this particular class.
 */
@property(nonatomic, assign) float *_Nullable *_Nullable confidenceMasks;

/** Flattened 2D-array of size `width` x `height`, in row major order.
 * The value of each pixel in this mask represents the class to which the
 * pixel belongs.
 */
@property(nonatomic, assign, nullable) NSUInteger *categoryMask;

/** Number of colored labels which is equivalent to number of classes
 * supported by the model.
 */
@property(nonatomic, assign) NSUInteger coloredLabelsSize;

/**
 * The list of colored labels for all the supported categories (classes).
 * Depending on which is present, this list is in 1:1 correspondence with:
 * `category_mask` pixel values, i.e. a pixel with value `i` is associated with
 * `colored_labels[i]`, `confidence_masks` indices, i.e. `confidence_masks[i]`
 * is associated with `colored_labels[i]`.
 */
@property(nonatomic, copy) NSArray<TFLColoredLabel *> *coloredLabels;

@end

/** Encapsulates results of any image segmentation task. */
@interface TFLSegmentationResult : NSObject

/** Array of segmentations returned after inference by model.
 * Note that at the time, this array is expected to have a single
 * `TfLiteSegmentation`; the field is made an array for later extension to
 * e.g. instance segmentation models, which may return one segmentation per
 * object.
 */
@property(nonatomic, copy) NSArray<TFLSegmentation *> *segmentations;

@end

NS_ASSUME_NONNULL_END
