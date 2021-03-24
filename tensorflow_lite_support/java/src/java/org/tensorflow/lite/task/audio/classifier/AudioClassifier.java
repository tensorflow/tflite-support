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

package org.tensorflow.lite.task.audio.classifier;

import org.tensorflow.lite.task.core.BaseTaskApi;

/**
 * Performs classification on {@link TensorAudio}.
 *
 * <p>IMPORTANT: This class is under developerment and the API is subjected to changes.
 */
public final class AudioClassifier extends BaseTaskApi {

  /**
   * Constructor to initialize the JNI with a pointer from C++.
   *
   * @param nativeHandle a pointer referencing memory allocated in C++
   */
  private AudioClassifier(long nativeHandle) {
    super(nativeHandle);
  }

  /**
   * Releases memory pointed by the pointer in the native layer.
   *
   * <p>TODO(b/183343074): This method is being implemented. For now, it does nothing.
   *
   * @param nativeHandle pointer to memory allocated
   */
  @Override
  protected void deinit(long nativeHandle) {
    // TODO(b/183343074): Implement this method.
  }
}
