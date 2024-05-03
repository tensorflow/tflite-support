/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.task.vision.embedder;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Rect;
import android.os.ParcelFileDescriptor;
import com.google.android.odml.image.MlImage;
import com.google.auto.value.AutoValue;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.List;
import org.tensorflow.lite.support.image.MlImageAdapter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.BaseOptions;
import org.tensorflow.lite.task.core.TaskJniUtils;
import org.tensorflow.lite.task.core.TaskJniUtils.EmptyHandleProvider;
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions;
import org.tensorflow.lite.task.processor.FeatureVector;
import org.tensorflow.lite.task.processor.Embedding;
import org.tensorflow.lite.task.processor.EmbedderOptions;
import org.tensorflow.lite.task.vision.core.BaseVisionTaskApi;
import org.tensorflow.lite.task.vision.core.BaseVisionTaskApi.InferenceProvider;
import org.tensorflow.lite.task.core.TaskJniUtils.FdAndOptionsHandleProvider;
import org.tensorflow.lite.task.core.annotations.UsedByReflection;

/**
 * Performs embedding on images.
 *
 * <p>The API expects a TFLite model with optional, but strongly recommended, <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>.
 *
 * <ul>
 *   <li>Input image tensor ({@code kTfLiteUInt8}/{@code kTfLiteFloat32})
 *       <ul>
 *         <li>image input of size {@code [batch x height x width x channels]}.
 *         <li>batch inference is not supported ({@code batch} is required to be 1).
 *         <li>only RGB inputs are supported ({@code channels} is required to be 3).
 *         <li>if type is {@code kTfLiteFloat32}, NormalizationOptions are required to be attached
 *             to the metadata for input normalization.
 *       </ul>
 *   <li>Output tensor ({@code kTfLiteUInt8}/{@code kTfLiteFloat32})
 *       <ul>
 *         <li>{@code N} components corresponding to the {@code N} dimensions of the returned
 *             feature vector for this output layer.
 *         <li>Either 2 or 4 dimensions, i.e. {@code [1 x N]} or {@code [1 x 1 x 1 x N]}.
 *       </ul>
 * </ul>
 *
 * <p>TODO(b/180502532): add pointer to example model.
 *
 * <p>TODO(b/222671076): add factory create methods without options, such as `createFromFile`, once
 * the single file format (index file packed in the model) is supported.
 */
public final class ImageEmbedder extends BaseVisionTaskApi {

  private static final String IMAGE_EMBEDDER_NATIVE_LIB = "task_vision_jni";
  private static final int OPTIONAL_FD_LENGTH = -1;
  private static final int OPTIONAL_FD_OFFSET = -1;

  /**
   * Creates an {@link ImageEmbedder} instance from the default {@link ImageEmbedderOptions}.
   *
   * @param modelPath path to the embed model with metadata in the assets
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws IllegalArgumentException if an argument is invalid
   * @throws IllegalStateException if there is an internal error
   * @throws RuntimeException if there is an otherwise unspecified error
   */
  public static ImageEmbedder createFromFile(Context context, String modelPath)
      throws IOException {
    return createFromFileAndOptions(context, modelPath, ImageEmbedderOptions.builder().build());
  }

  /**
   * Creates an {@link ImageEmbedder} instance from the default {@link ImageEmbedderOptions}.
   *
   * @param modelFile the embed model {@link File} instance
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws IllegalArgumentException if an argument is invalid
   * @throws IllegalStateException if there is an internal error
   * @throws RuntimeException if there is an otherwise unspecified error
   */
  public static ImageEmbedder createFromFile(File modelPath) throws IOException {
    return createFromFileAndOptions(modelPath, ImageEmbedderOptions.builder().build());
  }

  /**
   * Creates an {@link ImageEmbedder} instance with a model buffer and the default {@link
   * ImageEmbedderOptions}.
   *
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the embed
   *     model
   * @throws IllegalArgumentException if the model buffer is not a direct {@link ByteBuffer} or a
   *     {@link MappedByteBuffer} * @throws IllegalStateException if there is an internal error
   * @throws RuntimeException if there is an otherwise unspecified error
   */
  public static ImageEmbedder createFromBuffer(final ByteBuffer modelBuffer) {
    return createFromBufferAndOptions(modelBuffer, ImageEmbedderOptions.builder().build());
  }

  /**
   * Creates an {@link ImageEmbedder} instance from {@link ImageEmbedderOptions}.
   *
   * @param modelPath path to the embed model with metadata in the assets
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws IllegalArgumentException if an argument is invalid
   * @throws IllegalStateException if there is an internal error
   * @throws RuntimeException if there is an otherwise unspecified error
   */
  public static ImageEmbedder createFromFileAndOptions(
      Context context, String modelPath, ImageEmbedderOptions options) throws IOException {
    return new ImageEmbedder(
        TaskJniUtils.createHandleFromFdAndOptions(
            context,
            new FdAndOptionsHandleProvider<ImageEmbedderOptions>() {
              @Override
              public long createHandle(
                  int fileDescriptor,
                  long fileDescriptorLength,
                  long fileDescriptorOffset,
                  ImageEmbedderOptions options) {
                return initJniWithModelFdAndOptions(
                    fileDescriptor,
                    fileDescriptorLength,
                    fileDescriptorOffset,
                    options,
                    TaskJniUtils.createProtoBaseOptionsHandleWithLegacyNumThreads(
                        options.getBaseOptions(), options.getNumThreads()));
              }
            },
            IMAGE_EMBEDDER_NATIVE_LIB,
            modelPath,
            options));
  }

  /**
   * Creates an {@link ImageEmbedder} instance from {@link ImageEmbedderOptions}.
   *
   * @param modelFile the embed model {@link File} instance
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws IllegalArgumentException if an argument is invalid
   * @throws IllegalStateException if there is an internal error
   * @throws RuntimeException if there is an otherwise unspecified error
   */
  public static ImageEmbedder createFromFileAndOptions(
      File modelFile, final ImageEmbedderOptions options) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      return new ImageEmbedder(
          TaskJniUtils.createHandleFromLibrary(
              new TaskJniUtils.EmptyHandleProvider() {
                @Override
                public long createHandle() {
                  return initJniWithModelFdAndOptions(
                      descriptor.getFd(),
                      /*fileDescriptorLength=*/ OPTIONAL_FD_LENGTH,
                      /*fileDescriptorOffset=*/ OPTIONAL_FD_OFFSET,
                      options,
                      TaskJniUtils.createProtoBaseOptionsHandleWithLegacyNumThreads(
                          options.getBaseOptions(), options.getNumThreads()));
                }
              },
              IMAGE_EMBEDDER_NATIVE_LIB));
    }
  }

  /**
   * Creates an {@link ImageEmbedder} instance with a model buffer and {@link
   * ImageEmbedderOptions}.
   *
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the embed
   *     model
   * @throws IllegalArgumentException if the model buffer is not a direct {@link ByteBuffer} or a
   *     {@link MappedByteBuffer}
   * @throws IllegalStateException if there is an internal error
   * @throws RuntimeException if there is an otherwise unspecified error
   */
  public static ImageEmbedder createFromBufferAndOptions(
      final ByteBuffer modelBuffer, final ImageEmbedderOptions options) {
    if (!(modelBuffer.isDirect() || modelBuffer instanceof MappedByteBuffer)) {
      throw new IllegalArgumentException(
          "The model buffer should be either a direct ByteBuffer or a MappedByteBuffer.");
    }
    return new ImageEmbedder(
        TaskJniUtils.createHandleFromLibrary(
            new EmptyHandleProvider() {
              @Override
              public long createHandle() {
                return initJniWithByteBuffer(
                    modelBuffer,
                    options,
                    TaskJniUtils.createProtoBaseOptionsHandleWithLegacyNumThreads(
                        options.getBaseOptions(), options.getNumThreads()));
              }
            },
            IMAGE_EMBEDDER_NATIVE_LIB));
  }

  /**
   * Constructor to initialize the JNI with a pointer from C++.
   *
   * @param nativeHandle a pointer referencing memory allocated in C++
   */
  ImageEmbedder(long nativeHandle) {
    super(nativeHandle);
  }

  /** Options for setting up an ImageEmbedder. */
  @UsedByReflection("image_embedder_jni.cc")
  public static class ImageEmbedderOptions {
    private final BaseOptions baseOptions;
    private final boolean l2Normalize;
    private final boolean quantize;
    private final int numThreads;

    public static Builder builder() {
      return new Builder();
    }

    /** A builder that helps to configure an instance of ObjectDetectorOptions. */
    public static class Builder {
      private BaseOptions baseOptions = BaseOptions.builder().build();
      private boolean l2Normalize = true;
      private boolean quantize = false;
      private int numThreads = -1;

      private Builder() {}

      /** Sets the general options to configure Task APIs, such as accelerators. */
      public Builder setBaseOptions(BaseOptions baseOptions) {
        this.baseOptions = baseOptions;
        return this;
      }

      /**
       * Sets the l2 normalization that overrides the one provided in the model metadata (if any).
       * Results below this value are rejected.
       */
      public Builder setL2Normalize(boolean l2Normalize) {
        this.l2Normalize = l2Normalize;
        return this;
      }

      /**
       * Sets the quantization that overrides the one provided in the model metadata (if any).
       * Results below this value are rejected.
       */
      public Builder setQuantize(boolean quantize) {
        this.quantize = quantize;
        return this;
      }

      /**
       * Sets the number of threads to be used for TFLite ops that support multi-threading when
       * running inference with CPU. Defaults to -1.
       *
       * <p>numThreads should be greater than 0 or equal to -1. Setting numThreads to -1 has the
       * effect to let TFLite runtime set the value.
       *
       * @deprecated use {@link BaseOptions} to configure number of threads instead. This method
       *     will override the number of threads configured from {@link BaseOptions}.
       */
      @Deprecated
      public Builder setNumThreads(int numThreads) {
        this.numThreads = numThreads;
        return this;
      }

      public ImageEmbedderOptions build() {
        return new ImageEmbedderOptions(this);
      }
    }

    @UsedByReflection("image_embedder_jni.cc")
    public boolean getL2Normalize() {
      return l2Normalize;
    }

    @UsedByReflection("image_embedder_jni.cc")
    public boolean getQuantize() {
      return quantize;
    }

    @UsedByReflection("image_embedder_jni.cc")
    public int getNumThreads() {
      return numThreads;
    }

    public BaseOptions getBaseOptions() {
      return baseOptions;
    }

    private ImageEmbedderOptions(Builder builder) {
      l2Normalize = builder.l2Normalize;
      quantize = builder.quantize;
      numThreads = builder.numThreads;
      baseOptions = builder.baseOptions;
    }
  }

  /**
   * Performs embedding extraction on the provided {@link TensorImage}
   *
   * <p>{@link ImageEmbedder} supports the following {@link TensorImage} color space types:
   *
   * <ul>
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#RGB}
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#NV12}
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#NV21}
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#YV12}
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#YV21}
   * </ul>
   *
   * @param image a UINT8 {@link TensorImage} object that represents an RGB or YUV image
   * @throws IllegalArgumentException if the color space type of image is unsupported
   */
  public List<Embedding> embed(TensorImage image) {
    return embed(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs embedding extraction on the provided {@link TensorImage} with {@link
   * ImageProcessingOptions}.
   *
   * <p>{@link ImageEmbedder} supports the following options:
   *
   * <ul>
   *   <li>Region of interest (ROI) (through {@link ImageProcessingOptions.Builder#setRoi}). It
   *       defaults to the entire image.
   *   <li>image rotation (through {@link ImageProcessingOptions.Builder#setOrientation}). It
   *       defaults to {@link ImageProcessingOptions.Orientation#TOP_LEFT}.
   * </ul>
   *
   * <p>{@link ImageEmbedder} supports the following {@link TensorImage} color space types:
   *
   * <ul>
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#RGB}
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#NV12}
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#NV21}
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#YV12}
   *   <li>{@link org.tensorflow.lite.support.image.ColorSpaceType#YV21}
   * </ul>
   *
   * @param image a UINT8 {@link TensorImage} object that represents an RGB or YUV image
   * @throws IllegalArgumentException if the color space type of image is unsupported
   */
  public List<Embedding> embed(TensorImage image, ImageProcessingOptions options) {
    return run(
        new InferenceProvider<List<Embedding>>() {
          @Override
          public List<Embedding> run(
              long frameBufferHandle, int width, int height, ImageProcessingOptions options) {
            return embed(frameBufferHandle, width, height, options);
          }
        },
        image,
        options);
  }

  /**
   * Performs embedding extraction on the provided {@code MlImage}
   *
   * @param image an {@code MlImage} object that represents an image
   * @throws IllegalArgumentException if the storage type or format of the image is unsupported
   */
  public List<Embedding> embed(MlImage image) {
    return embed(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs embedding extraction on the provided {@code MlImage} with {@link
   * ImageProcessingOptions}
   *
   * <p>{@link ImageEmbedder} supports the following options:
   *
   * <ul>
   *   <li>Region of interest (ROI) (through {@link ImageProcessingOptions.Builder#setRoi}). It
   *       defaults to the entire image.
   *   <li>image rotation (through {@link ImageProcessingOptions.Builder#setOrientation}). It
   *       defaults to {@link ImageProcessingOptions.Orientation#TOP_LEFT}. {@link
   *       MlImage#getRotation()} is not effective.
   * </ul>
   *
   * @param image a {@code MlImage} object that represents an image
   * @param options configures options including ROI and rotation
   * @throws IllegalArgumentException if the storage type or format of the image is unsupported
   */
  public List<Embedding> embed(MlImage image, ImageProcessingOptions options) {
    image.getInternal().acquire();
    TensorImage tensorImage = MlImageAdapter.createTensorImageFrom(image);
    List<Embedding> result = embed(tensorImage, options);
    image.close();
    return result;
  }

  private List<Embedding> embed(
      long frameBufferHandle, int width, int height, ImageProcessingOptions options) {
    checkNotClosed();
    Rect roi = options.getRoi().isEmpty() ? new Rect(0, 0, width, height) : options.getRoi();
    return embedNative(
        getNativeHandle(),
        frameBufferHandle,
        new int[] {roi.left, roi.top, roi.width(), roi.height()});
  }

  private static native long initJniWithModelFdAndOptions(
      int fileDescriptor,
      long fileDescriptorLength,
      long fileDescriptorOffset,
      ImageEmbedderOptions options,
      long baseOptionsHandle);

  private static native long initJniWithByteBuffer(
      ByteBuffer modelBuffer, ImageEmbedderOptions options, long baseOptionsHandle);

  private static native List<Embedding> detectNative(long nativeHandle, long frameBufferHandle);

  /**
   * The native method to embed an image based on the ROI specified.
   *
   * @param roi the ROI of the input image, an array representing the bounding box as {left, top,
   *     width, height}
   */
  private static native List<Embedding> embedNative(
      long nativeHandle, long frameBufferHandle, int[] roi);

  @Override
  protected void deinit(long nativeHandle) {
    deinitJni(nativeHandle);
  }

  /**
   * Native implementation to release memory pointed by the pointer.
   *
   * @param nativeHandle pointer to memory allocated
   */
  private native void deinitJni(long nativeHandle);
}
