/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.support.tensorbuffer;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.internal.SupportPreconditions;

/** Represents data buffer with 8-bit unsigned integer values. */
public final class TensorBufferUint8 extends TensorBuffer {
  private static final DataType DATA_TYPE = DataType.UINT8;

  /**
   * Creates a {@link TensorBufferUint8} with specified {@code shape}.
   *
   * @throws NullPointerException if {@code shape} is null.
   * @throws IllegalArgumentException if {@code shape} has non-positive elements.
   */
  TensorBufferUint8(@NonNull int[] shape) {
    super(shape);
  }

  TensorBufferUint8() {
    super();
  }

  @Override
  public DataType getDataType() {
    return DATA_TYPE;
  }

  @Override
  @NonNull
  public float[] getFloatArray() {
    buffer.rewind();
    byte[] byteArr = new byte[flatSize];
    buffer.get(byteArr);

    float[] floatArr = new float[flatSize];
    for (int i = 0; i < flatSize; i++) {
      floatArr[i] = (float) (byteArr[i] & 0xff);
    }
    return floatArr;
  }

  @Override
  public float getFloatValue(int index) {
    return (float) (buffer.get(index) & 0xff);
  }

  @Override
  @NonNull
  public int[] getIntArray() {
    buffer.rewind();
    byte[] byteArr = new byte[flatSize];
    buffer.get(byteArr);

    int[] intArr = new int[flatSize];
    for (int i = 0; i < flatSize; i++) {
      intArr[i] = byteArr[i] & 0xff;
    }
    return intArr;
  }

  @Override
  public int getIntValue(int index) {
    return buffer.get(index) & 0xff;
  }

  @Override
  public int getTypeSize() {
    return DATA_TYPE.byteSize();
  }

  @Override
  public void loadArray(@NonNull float[] src, @NonNull int[] shape) {
    SupportPreconditions.checkNotNull(src, "The array to be loaded cannot be null.");
    SupportPreconditions.checkArgument(
        src.length == computeFlatSize(shape),
        "The size of the array to be loaded does not match the specified shape.");
    copyByteBufferIfReadOnly();
    resize(shape);
    buffer.rewind();

    byte[] byteArr = new byte[src.length];
    int cnt = 0;
    for (float a : src) {
      byteArr[cnt++] = (byte) Math.max(Math.min(a, 255.0), 0.0);
    }
    buffer.put(byteArr);
  }

  @Override
  public void loadArray(@NonNull int[] src, @NonNull int[] shape) {
    SupportPreconditions.checkNotNull(src, "The array to be loaded cannot be null.");
    SupportPreconditions.checkArgument(
        src.length == computeFlatSize(shape),
        "The size of the array to be loaded does not match the specified shape.");
    copyByteBufferIfReadOnly();
    resize(shape);
    buffer.rewind();

    byte[] byteArr = new byte[src.length];
    int cnt = 0;
    for (float a : src) {
      byteArr[cnt++] = (byte) Math.max(Math.min(a, 255), 0);
    }
    buffer.put(byteArr);
  }
}
