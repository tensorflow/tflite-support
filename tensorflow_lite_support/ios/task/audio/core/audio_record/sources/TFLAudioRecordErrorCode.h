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

/**
 * @enum TFLAudioRecordErrorCode
 * This enum specifies  error codes for TFLAudioRecord of the TensorFlow Lite Task Library.
 */
typedef NS_ENUM(NSUInteger, TFLAudioRecordErrorCode) {

  /** Unspecified error. */
  TFLAudioRecordErrorCodeUnspecifiedError = 1,

  /** Invalid argument specified. */
  TFLAudioRecordErrorCodeInvalidArgumentError,

  /**
   * Audio processing operation failed.
   * E.g. Format conversion operations by TFLAudioRecord.
   */
  TFLAudioRecordErrorCodeProcessingError,

  /**
   * Audio record permissions were denied by the user.
   */
  TFLAudioRecordErrorCodeRecordPermissionDeniedError,

  /**
   * Audio record permissions cannot be determined. If this error is returned by
   * TFLAudioRecord, the caller has to acquire permissions using AVFoundation.
   */
  TFLAudioRecordErrorCodeRecordPermissionUndeterminedError,

  /**
   * TFLAudioRecord is waiting for new mic input.
   */
  TFLAudioRecordErrorCodeWaitingForNewMicInputError

} NS_SWIFT_NAME(AudioRecordErrorCode);

NS_ASSUME_NONNULL_END
