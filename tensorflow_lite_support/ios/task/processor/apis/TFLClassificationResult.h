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
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/** Encapsulates information about a class in the classification results. */
@interface TFLCategory : NSObject

/** Display name of the class. */
@property(nonatomic, copy) NSString *displayName;

/** Class name of the class . */
@property(nonatomic, copy) NSString *label;

 /** Confidence score for this class . */
@property(nonatomic, assign) float score;

/** The index of the class in the corresponding label map, usually packed in the TFLite Model Metadata. */
@property(nonatomic, assign) NSInteger classIndex;

@end

/** Encapsulates list of predicted classes (aka labels) for a given image classifier head. */
@interface TFLClassifications : NSObject

/**
 * The index of the image classifier head these classes refer to. This is useful for multi-head models.
 */
@property(nonatomic, assign) int headIndex;

/** The array of predicted classes, usually sorted by descending scores (e.g.from high to low probability). */
@property(nonatomic, copy) NSArray<TFLCategory *> *categories;

/**
 * Initializes an instance of TFLClassifications for an image classifier head with given index and list of
 * predicted categories.
 *
 * @param categories list of predicted categories for classification head that should be represeented by
 * the initialized TFLClassification.
 * @param headIndex index of the image classifier head this instance of TFLClassifications should
 * represent.
 * @return An instance of TFLClassifications.
 */
- (instancetype)initWithCategories:(NSArray<TFLCategory *> *)categories headIndex:(int)headIndex;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

/** Encapsulates results of any classification task. */
@interface TFLClassificationResult : NSObject

@property(nonatomic, copy) NSArray<TFLClassifications *> *classifications;

/**
 * Initializes an instance of TFLClassifications for an image classifier head with given index and list of
 * predicted categories.
 *
 * @param classifications list containing results of image classifier heads.
 * @return An instance of TFLClassifications.
 */
- (instancetype)initWithClassifications:(NSArray<TFLClassifications *> *)classifications;

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
