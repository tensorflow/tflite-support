//
//  TFLBaseOptions.h
//  TFLTaskImageClassifierFramework
//
//  Created by Prianka Kariat on 07/09/21.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * Holds cpu settings.
 */
NS_SWIFT_NAME(CpuSettings)
@interface TFLCpuSettings : NSObject

/** Specifies the number of threads to be used for TFLite ops that support multi-threadingwhen
 * running inference with CPU.
 * @discussion This property hould be greater than 0 or equal to -1. Setting  it to -1 has the
 * effect to let TFLite runtime set the value.
 */
@property(nonatomic, assign) int numThreads;

@end

/**
 * Holds settings for one possible acceleration configuration.
 */
NS_SWIFT_NAME(ComputeSettings)
@interface TFLComputeSettings : NSObject

/** Holds cpu settings. */
@property(nonatomic, strong) TFLCpuSettings *cpuSettings;

@end

/**
 * Holds settings for one possible acceleration configuration.
 */
NS_SWIFT_NAME(ExternalFile)
@interface TFLExternalFile : NSObject

/** Path to the file in bundle. */
@property(nonatomic, strong) NSString *filePath;
/// Add provision for other sources in future.

@end

/**
 * Holds the base options that is used for creation of any type of task. It has fields with
 * important information acceleration configuration, tflite model source etc.
 */
NS_SWIFT_NAME(BaseOptions)
@interface TFLBaseOptions : NSObject

/**
 * The external model file, as a single standalone TFLite file. It could be packed with TFLite Model
 * Metadata[1] and associated files if exist. Fail to provide the necessary metadata and associated
 * files might result in errors.
 */
@property(nonatomic, strong) TFLExternalFile *modelFile;

/**
 * Holds settings for one possible acceleration configuration including.cpu/gpu settings.
 * Please see documentation of TfLiteComputeSettings and its members for more details.
 */
@property(nonatomic, strong) TFLComputeSettings *computeSettings;

@end

NS_ASSUME_NONNULL_END
