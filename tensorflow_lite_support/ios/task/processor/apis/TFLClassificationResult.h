//
//  TFLClassificationResult.h
//  TFLTaskImageClassifierFramework
//
//  Created by Prianka Kariat on 07/09/21.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/** Encapsulates information about a class in the classification results. */
@interface TFLCategory : NSObject

/** Display name of the class. */
@property(nonatomic, strong) NSString *displayName;

/** Class name of the class . */
@property(nonatomic, strong) NSString *label;

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
@property(nonatomic, strong) NSArray<TFLCategory *> *categories;

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

@property(nonatomic, strong) NSArray<TFLClassifications *> *classifications;

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
