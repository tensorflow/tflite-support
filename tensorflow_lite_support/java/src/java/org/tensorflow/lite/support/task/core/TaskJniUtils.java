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
import android.content.res.AssetFileDescriptor;
import android.util.Log;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/** JNI utils for Task API. */
public class TaskJniUtils {
  public static final long INVALID_POINTER = 0;
  private static final String TAG = TaskJniUtils.class.getSimpleName();
  /** Syntax sugar to get nativeHandle from empty param list. */
  public interface EmptyHandleProvider {
    long createHandle();
  }

  /** Syntax sugar to get nativeHandle from an array of {@link ByteBuffer}s. */
  public interface MultipleBuffersHandleProvider {
    long createHandle(ByteBuffer... buffers);
  }

  /**
   * Initializes the JNI and returns C++ handle by first loading the C++ library and then invokes
   * {@link EmptyHandleProvider#createHandle()}.
   *
   * @param provider provider to get C++ handle, usually returned from native call
   * @return C++ handle as long
   */
  static long createHandleFromLibrary(EmptyHandleProvider provider, String libName) {
    tryLoadLibrary(libName);
    try {
      return provider.createHandle();
    } catch (Exception e) {
      String errorMessage = "Error getting native address of native library: " + libName;
      Log.e(TAG, errorMessage, e);
      throw new IllegalStateException(errorMessage, e);
    }
  }

  /**
   * Initializes the JNI and returns C++ handle by first loading the C++ library and then invokes
   * {@link MultipleBuffersHandleProvider#createHandle(ByteBuffer...)}.
   *
   * @param context app context
   * @param provider provider to get C++ pointer, usually returned from native call
   * @param libName name of C++ lib to load
   * @param filePaths file paths to load
   * @return C++ pointer as long
   */
  public static long createHandleWithMultipleAssetFilesFromLibrary(
      Context context,
      MultipleBuffersHandleProvider provider,
      String libName,
      String... filePaths) {
    try {
      MappedByteBuffer[] buffers = new MappedByteBuffer[filePaths.length];
      for (int i = 0; i < filePaths.length; i++) {
        buffers[i] = loadMappedFile(context, filePaths[i]);
      }
      return createHandleFromLibrary(() -> provider.createHandle(buffers), libName);
    } catch (Exception e) {
      String errorMessage =
          "Error getting native address of native library: " + libName + " from modelPaths";
      Log.e(TAG, errorMessage, e);
      throw new IllegalStateException(errorMessage, e);
    }
  }

  /**
   * Loads a file from the asset folder through memory mapping.
   *
   * @param context Application context to access assets.
   * @param filePath Asset path of the file.
   * @return the loaded memory mapped file.
   * @throws IOException if an I/O error occurs when loading the tflite model.
   */
  static MappedByteBuffer loadMappedFile(Context context, String filePath) throws IOException {
    try (AssetFileDescriptor fileDescriptor = context.getAssets().openFd(filePath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  private TaskJniUtils() {}

  /**
   * Try load a native library, if it's already loaded return directly.
   *
   * @param libName name of the lib
   */
  private static void tryLoadLibrary(String libName) {
    try {
      System.loadLibrary(libName);
    } catch (UnsatisfiedLinkError e) {
      String errorMessage = "Error loading native library: " + libName;
      Log.e(TAG, errorMessage, e);
      throw new UnsatisfiedLinkError(errorMessage);
    }
  }
}
