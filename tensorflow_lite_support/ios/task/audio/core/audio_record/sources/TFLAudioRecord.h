// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import <Foundation/Foundation.h>
#import "tensorflow_lite_support/ios/task/audio/core/sources/TFLAudioFormat.h"
#import "tensorflow_lite_support/ios/task/audio/core/sources/TFLFloatBuffer.h"

NS_ASSUME_NONNULL_BEGIN

/** A wrapper class to tap the device's microphone continuously. Currently this class only supports
 tapping the input node of the AVAudioEngine which emits audio data having only one channel.*/
NS_SWIFT_NAME(AudioRecord)
@interface TFLAudioRecord : NSObject

/** Audio format specifying the number of channels and sample rate supported. */
@property(nonatomic, readonly) TFLAudioFormat *audioFormat;

/** Size of the buffer held by `TFLAudioRecord`. It ensures delivery of audio data of length
 * bufferSize arrays when you tap the input microphone. */
@property(nonatomic, readonly) NSUInteger bufferSize;

/**
 * Initializes a new `TFLAudioRecord` with a given `TFLAudioFormat` and buffer size.
 *
 * @param format An audio format of type `TFLAudioFormat`.
 * @param bufferSize Maximum number of elements the internal buffer of `TFLAudioRecord` can hold at any given point of time. The buffer length should be a multiple of format.channelCount.
 * @param error An optional error parameter populated if the initialization of `TFLAudioRecord` was not successful.
 *
 * @return An new instance of `TFLAudioRecord`. `nil` if there is an error in initializing `TFLAudioRecord`.
 */
- (nullable instancetype)initWithAudioFormat:(TFLAudioFormat *)format
                                  bufferSize:(NSUInteger)bufferSize
                                       error:(NSError *_Nullable *)error;

/**
 * This function starts tapping the input audio samples from the mic if audio record permissions
 * have been granted by the user. 
 * 
 * @discussion Before calling this function, you must call
 * - [AVAudioSession requestRecordPermission:] on [AVAudioSession sharedInstance] to acquire
 * record permissions. If the user has denied permission or the permissions are undetermined, the return value will be false and
 * appropriate error is populated in the error pointer.
 * The internal buffer of TFLAudioRecord of length bufferSize will
 * always have the most recent data samples acquired from the mic if this function returns successfully.  Use:
 * - [TFLAudioRecord readAtOffset:withSize:error:] to get the data from the buffer at any instance, if audio recording has
 * started successfully.
 *
 * Use - [TFLAudioRecord stop] to stop tapping the  mic input.
 *
 * @param error An optional error parameter populated when the mic input could not be tapped successfully.
 * 
 * @return Boolean value indicating if audio recording started successfully. If NO, and an address
 * to an error is passed in, the error will hold the reason for failure once the function returns.
 */
- (BOOL)startRecordingWithError:(NSError **)error NS_SWIFT_NAME(startRecording());

/**
 * Stops tapping the audio samples from the input mic. All elements in the internal buffer of `TFLAudioRecord` will also be set to zero.
 */
- (void)stop;

/**
 * Returns the size number of elements in the internal buffer of `TFLAudioRecord` starting at offset, i.e, buffer[offset:offset+size].
 *
 * @param offset Index in the buffer from which elements are to be read.
 * @param size Number of elements to be returned.
 * @param error An optional error parameter populated if the internal buffer could not be read successfully.
 *
 * @returns A `TFLFloatBuffer` containing the elements of the internal buffer of `TFLAudioRecord` in the range, i.e, buffer[offset:offset+size].
 * `nil` if there is an error in reading the internal buffer.
 */
- (nullable TFLFloatBuffer *)readAtOffset:(NSUInteger)offset
                                 withSize:(NSUInteger)size
                                    error:(NSError *_Nullable *)error;

@end

NS_ASSUME_NONNULL_END
