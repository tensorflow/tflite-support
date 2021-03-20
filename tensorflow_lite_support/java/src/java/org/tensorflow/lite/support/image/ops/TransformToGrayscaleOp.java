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

package org.tensorflow.lite.support.image.ops;

import static org.tensorflow.lite.support.common.SupportPreconditions.checkArgument;
import static org.tensorflow.lite.support.image.ColorSpaceType.GRAYSCALE;

import android.graphics.Canvas;
import android.graphics.Bitmap;
import android.graphics.Paint;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.ColorFilter;
import android.graphics.PointF;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.support.image.ColorSpaceType;
import org.tensorflow.lite.support.image.ImageOperator;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.DataType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Transforms an RGB image to GrayScale as a image processing unit.
 * <p>
 * The conversion is based on OpenCV RGB to GRAY conversion
 * https://docs.opencv.org/master/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
 * <p>
 * The 3 channels of the grayscale image are identical and one of them is used to create the TensorBuffer
 */
public class TransformToGrayscaleOp implements ImageOperator {

  private final float[] matrix;

  /**
   * Creates a TransformToGrayscaleOp.
   */
  public TransformToGrayscaleOp() {
    // A matrix is created that will be applied later to canvas to generate grayscale image
    // The luminance of each pixel is calculated as the weighted sum of the 3 RGB values
    // Y = 0.299R + 0.587G + 0.114B
    this.matrix = new float[]{0.299F, 0.587F, 0.114F, 0.0F, 0.0F,
            0.299F, 0.587F, 0.114F, 0.0F, 0.0F,
            0.299F, 0.587F, 0.114F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
  }

  /**
   * Applies the defined transformation to grayscale and returns a TensorImage.
   * <p>
   * Because the 3 channels of the image are identical only one is used to create the TensorBuffer
   * <p>
   * In case the ColorSpaceType of the TensorImage is GRAYSCALE it skips and returns the same image,
   * otherwise it checks for RGB type of the TensorImage and proceeds.
   *
   * @param image input image.
   * @return output image.
   */
  @Override
  @NonNull
  public TensorImage apply(@NonNull TensorImage image) {
    if (image.getColorSpaceType() == GRAYSCALE) {
      return image;
    } else {
      checkArgument(
              image.getColorSpaceType() == ColorSpaceType.RGB,
              "Only RGB images are supported in TransformToGrayscaleOp, but not " + image.getColorSpaceType().name());
    }
    int height = image.getBitmap().getHeight();
    int width = image.getBitmap().getWidth();
    Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
    Canvas canvas = new Canvas(bmpGrayscale);
    Paint paint = new Paint();
    ColorMatrixColorFilter colorMatrixFilter = new ColorMatrixColorFilter(matrix);
    paint.setColorFilter((ColorFilter) colorMatrixFilter);
    canvas.drawBitmap(image.getBitmap(), 0.0F, 0.0F, paint);

    // Get the pixels from the generated grayscale image
    int w = bmpGrayscale.getWidth();
    int h = bmpGrayscale.getHeight();
    int[] intValues = new int[w * h];
    bmpGrayscale.getPixels(intValues, 0, w, 0, 0, w, h);
    // Shape with one channel
    int[] shape = new int[]{1, h, w, 1};

    switch (image.getDataType()) {
      case UINT8:
        // Create byte array and use one of the 3 identical channels
        byte[] byteArr = new byte[w * h * 1];
        for (int i = 0, j = 0; i < intValues.length; i++) {
          byteArr[j++] = (byte) (intValues[i] & 0xff);
        }
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(1 * w * h * 1 * 1);
        byteBuffer.order(ByteOrder.nativeOrder());
        byteBuffer.put(byteArr);

        TensorBuffer buffer = TensorBuffer.createFixedSize(shape, DataType.UINT8);
        buffer.loadBuffer(byteBuffer, shape);

        image.load(buffer, GRAYSCALE);

        break;
      case FLOAT32:
        // Create float array and use one of the 3 identical channels
        float[] floatArr = new float[w * h * 1];
        for (int i = 0, j = 0; i < intValues.length; i++) {
          floatArr[j++] = (float) (intValues[i] & 0xff);
        }

        TensorBuffer bufferFloat = TensorBuffer.createFixedSize(shape, DataType.FLOAT32);
        bufferFloat.loadArray(floatArr, shape);

        image.load(bufferFloat, GRAYSCALE);

        break;
      default:
        // Should never happen.
        throw new IllegalStateException(
                "The type of TensorImage, " + image.getDataType() + ", is unsupported.");
    }

    return image;
  }

  @Override
  public int getOutputImageHeight(int inputImageHeight, int inputImageWidth) {
    return inputImageHeight;
  }

  @Override
  public int getOutputImageWidth(int inputImageHeight, int inputImageWidth) {
    return inputImageWidth;
  }

  @Override
  public PointF inverseTransform(PointF point, int inputImageHeight, int inputImageWidth) {
    return point;
  }
}
