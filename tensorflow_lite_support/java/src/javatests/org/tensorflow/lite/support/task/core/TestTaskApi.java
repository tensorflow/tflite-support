/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.support.task.core;

import android.content.Context;
import android.util.Log;
import java.nio.ByteBuffer;

/**
 * Sample subclass of {@link BaseTaskApi} for test.
 *
 * <p>The class loads a C++ binary from test_task_api_jni.cc and performs a simple add() function
 * that calls TestApi::Add from C++.
 *
 * <p>The class also demonstrates 2 ways to initialize and return the TestApi pointer from C++. One
 * without passing in any file, another with two files passed.
 */
public class TestTaskApi extends BaseTaskApi {
  private static final String TAG = TestTaskApi.class.getSimpleName();

  static final String TEST_LIB_NAME = "test_task_api_native_jni";

  TestTaskApi(long nativeHandle) {
    super(nativeHandle);
  }

  /** Public facing Java function, clients should only access this function to use the API. */
  public int add(int i1, int i2) {
    return addNative(getNativeHandle(), i1, i2);
  }

  /** Create an API instance without any parameters. */
  static TestTaskApi createTestTaskApi() {
    try {
      return new TestTaskApi(
          TaskJniUtils.createHandleFromLibrary(TestTaskApi::initJni, TEST_LIB_NAME));
    } catch (Exception e) {
      Log.e(TAG, "Failed to create TestTaskApi", e);
      return null;
    }
  }

  /** Create an API instance with required files. */
  static TestTaskApi createTestTaskApiWithBuffers(Context context, String file1, String file2) {
    return new TestTaskApi(
        TaskJniUtils.createHandleWithMultipleAssetFilesFromLibrary(
            context, TestTaskApi::initJniWithByteBuffers, TEST_LIB_NAME, file1, file2));
  }

  /** Invokes corresponding function in C++ and return results. */
  native int addNative(long nativeHandle, int i1, int i2);

  /** Initializes a TestApi instance in C++ and return its pointer. */
  static native long initJni();

  /** Initializes a TestApi instance with buffers in C++ and return its pointer. */
  static native long initJniWithByteBuffers(ByteBuffer... buffers);
}
