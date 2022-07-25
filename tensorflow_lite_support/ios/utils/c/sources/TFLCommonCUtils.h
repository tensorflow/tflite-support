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
#import "tensorflow_lite_support/ios/utils/sources/TFLCommonUtils.h"

#include "tensorflow_lite_support/c/common.h"

NS_ASSUME_NONNULL_BEGIN

/** Helper utility for the all tasks which encapsulates common functionality of iOS task library backed by C APIs. */
@interface TFLCommonUtils : TFLCommonUtils

/**
 * Converts a C library error, TfLiteSupportError to an NSError.
 *
 * @param supportError C library error.
 * @param error Pointer to the memory location where the created error should be saved. If `nil`,
 * no error will be saved.
 */
+ (BOOL)checkCError:(TfLiteSupportError *)supportError toError:(NSError **)error;

@end

NS_ASSUME_NONNULL_END
