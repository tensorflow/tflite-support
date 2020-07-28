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

package org.tensorflow.lite.support.image;

import static org.tensorflow.lite.support.common.SupportPreconditions.checkArgument;
import static org.tensorflow.lite.support.common.SupportPreconditions.checkState;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/** Holds a {@link TensorBuffer} and converts it to other image formats as needed. */
final class TensorBufferContainer implements ImageContainer {

  private final TensorBuffer buffer;
  private static final int HEIGHT_DIM = -3; // The third to last element of the shape.
  private static final int WIDTH_DIM = -2; // The second to last element of the shape.
  private static final int CHANNEL_DIM = -1; // The last element of the shape.
  private static final int BATCH_DIM =
      0; // The batch axis may not exist. But if it does, it is the first element of the shape.

  static TensorBufferContainer create(TensorBuffer buffer) {
    return new TensorBufferContainer(buffer);
  }

  private TensorBufferContainer(TensorBuffer buffer) {
    assertRGBImageShape(buffer.getShape());
    this.buffer = buffer;
  }

  // Verifies if the tensor shape is [h, w, 3] or [1, h, w, 3].
  static void assertRGBImageShape(int[] shape) {
    checkArgument(
        (shape.length == 3 || (shape.length == 4 && shape[BATCH_DIM] == 1))
            && getArrayElement(shape, HEIGHT_DIM) > 0
            && getArrayElement(shape, WIDTH_DIM) > 0
            && getArrayElement(shape, CHANNEL_DIM) == 3,
        "Only supports image shape in (h, w, c) or (1, h, w, c), and channels representing R, G, B"
            + " in order.");
  }

  @Override
  public TensorBufferContainer clone() {
    return create(TensorBuffer.createFrom(buffer, buffer.getDataType()));
  }

  @Override
  public Bitmap getBitmap() {
    checkState(
        buffer.getDataType() == DataType.UINT8,
        "TensorBufferContainer is holding a float-value image which is not able to convert to a"
            + " Bitmap.");

    Bitmap bitmap = Bitmap.createBitmap(getWidth(), getHeight(), Config.ARGB_8888);
    ImageConversions.convertTensorBufferToBitmap(buffer, bitmap);
    return bitmap;
  }

  @Override
  public TensorBuffer getTensorBuffer(DataType dataType) {
    // If the data type of buffer is desired, return it directly. Not making a defensive copy for
    // performance considerations. During image processing, users may need to set and get the
    // TensorBuffer many times.
    // Otherwise, create another one with the expected data type.
    return buffer.getDataType() == dataType ? buffer : TensorBuffer.createFrom(buffer, dataType);
  }

  @Override
  public int getWidth() {
    int[] shape = buffer.getShape();
    // The defensive check is needed, because buffer might be invalidly changed by users
    // (a.k.a internal data is corrupted)
    assertRGBImageShape(shape);
    return getArrayElement(shape, WIDTH_DIM);
  }

  @Override
  public int getHeight() {
    int[] shape = buffer.getShape();
    // The defensive check is needed, because buffer might be invalidly changed by users
    // (a.k.a internal data is corrupted)
    assertRGBImageShape(shape);
    return getArrayElement(shape, HEIGHT_DIM);
  }

  /**
   * Gets the element value through absolute indexing.
   *
   * @param index the index of the desired element. If negative, it will index relative to the end
   *     of the array. If index is out-of-bound, modulo will be applied.
   */
  private static int getArrayElement(int[] array, int index) {
    index = index % array.length;
    if (index < 0) {
      index += array.length;
    }
    return array[index];
  }
}
