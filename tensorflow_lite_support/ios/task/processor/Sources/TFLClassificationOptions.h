//
//  TFLClassificationOptions.h
//  TFLTaskImageClassifierFramework
//
//  Created by Prianka Kariat on 07/09/21.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Holds settings for any single classification task.
 */
@interface TFLClassificationOptions : NSObject

/** If set, all classes  in this list will be filtered out from the results . */
@property(nonatomic, strong) NSArray *labelDenyList;

/** If set, all classes not in this list will be filtered out from the results . */
@property(nonatomic, strong) NSArray *labelAllowList;

/** Display names local for display names*/
@property(nonatomic, strong) NSString *displayNamesLocal;

/** Results with score threshold greater than this value are returned . */
@property(nonatomic) float scoreThreshold;

/** Limit to the number of classes that can be returned in results. */
@property(nonatomic) NSInteger maxResults;

@end

NS_ASSUME_NONNULL_END
