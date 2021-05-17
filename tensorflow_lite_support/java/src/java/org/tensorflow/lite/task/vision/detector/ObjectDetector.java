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

package org.tensorflow.lite.task.vision.detector;

import android.content.Context;
import android.os.ParcelFileDescriptor;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.tensorflow.lite.annotations.UsedByReflection;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.core.TaskJniUtils;
import org.tensorflow.lite.task.core.TaskJniUtils.EmptyHandleProvider;
import org.tensorflow.lite.task.core.TaskJniUtils.FdAndOptionsHandleProvider;
import org.tensorflow.lite.task.core.vision.ImageProcessingOptions;
import org.tensorflow.lite.task.vision.core.BaseVisionTaskApi;
import org.tensorflow.lite.task.vision.core.BaseVisionTaskApi.InferenceProvider;

/**
 * Performs object detection on images.
 *
 * <p>The API expects a TFLite model with <a
 * href="https://www.tensorflow.org/lite/convert/metadata">TFLite Model Metadata.</a>.
 *
 * <p>The API supports models with one image input tensor and four output tensors. To be more
 * specific, here are the requirements.
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
 *   <li>Output tensors must be the 4 outputs of a {@code DetectionPostProcess} op, i.e:
 *       <ul>
 *         <li>Location tensor ({@code kTfLiteFloat32}):
 *             <ul>
 *               <li>tensor of size {@code [1 x num_results x 4]}, the inner array representing
 *                   bounding boxes in the form [top, left, right, bottom].
 *               <li>{@code BoundingBoxProperties} are required to be attached to the metadata and
 *                   must specify {@code type=BOUNDARIES} and {@code coordinate_type=RATIO}.
 *             </ul>
 *         <li>Classes tensor ({@code kTfLiteFloat32}):
 *             <ul>
 *               <li>tensor of size {@code [1 x num_results]}, each value representing the integer
 *                   index of a class.
 *               <li>if label maps are attached to the metadata as {@code TENSOR_VALUE_LABELS}
 *                   associated files, they are used to convert the tensor values into labels.
 *             </ul>
 *         <li>scores tensor ({@code kTfLiteFloat32}):
 *             <ul>
 *               <li>tensor of size {@code [1 x num_results]}, each value representing the score of
 *                   the detected object.
 *             </ul>
 *         <li>Number of detection tensor ({@code kTfLiteFloat32}):
 *             <ul>
 *               <li>integer num_results as a tensor of size {@code [1]}.
 *             </ul>
 *       </ul>
 * </ul>
 *
 * <p>An example of such model can be found on <a
 * href="https://tfhub.dev/google/lite-model/object_detection/mobile_object_localizer_v1/1/metadata/1">TensorFlow
 * Hub.</a>.
 */
public final class ObjectDetector extends BaseVisionTaskApi {

  private static final String OBJECT_DETECTOR_NATIVE_LIB = "task_vision_jni";
  private static final int OPTIONAL_FD_LENGTH = -1;
  private static final int OPTIONAL_FD_OFFSET = -1;

  /**
   * Creates an {@link ObjectDetector} instance from the default {@link ObjectDetectorOptions}.
   *
   * @param modelPath path to the detection model with metadata in the assets
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws AssertionError if error occurs when creating {@link ObjectDetector} from the native
   *     code
   */
  public static ObjectDetector createFromFile(Context context, String modelPath)
      throws IOException {
    return createFromFileAndOptions(context, modelPath, ObjectDetectorOptions.builder().build());
  }

  /**
   * Creates an {@link ObjectDetector} instance from the default {@link ObjectDetectorOptions}.
   *
   * @param modelFile the detection model {@link File} instance
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws AssertionError if error occurs when creating {@link ObjectDetector} from the native
   *     code
   */
  public static ObjectDetector createFromFile(File modelFile) throws IOException {
    return createFromFileAndOptions(modelFile, ObjectDetectorOptions.builder().build());
  }

  /**
   * Creates an {@link ObjectDetector} instance with a model buffer and the default {@link
   * ObjectDetectorOptions}.
   *
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the detection
   *     model
   * @throws AssertionError if error occurs when creating {@link ObjectDetector} from the native
   *     code
   * @throws IllegalArgumentException if the model buffer is not a direct {@link ByteBuffer} or a
   *     {@link MappedByteBuffer}
   */
  public static ObjectDetector createFromBuffer(final ByteBuffer modelBuffer) {
    return createFromBufferAndOptions(modelBuffer, ObjectDetectorOptions.builder().build());
  }

  /**
   * Creates an {@link ObjectDetector} instance from {@link ObjectDetectorOptions}.
   *
   * @param modelPath path to the detection model with metadata in the assets
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws AssertionError if error occurs when creating {@link ObjectDetector} from the native
   *     code
   */
  public static ObjectDetector createFromFileAndOptions(
      Context context, String modelPath, ObjectDetectorOptions options) throws IOException {
    return new ObjectDetector(
        TaskJniUtils.createHandleFromFdAndOptions(
            context,
            new FdAndOptionsHandleProvider<ObjectDetectorOptions>() {
              @Override
              public long createHandle(
                  int fileDescriptor,
                  long fileDescriptorLength,
                  long fileDescriptorOffset,
                  ObjectDetectorOptions options) {
                return initJniWithModelFdAndOptions(
                    fileDescriptor, fileDescriptorLength, fileDescriptorOffset, options);
              }
            },
            OBJECT_DETECTOR_NATIVE_LIB,
            modelPath,
            options));
  }

  /**
   * Creates an {@link ObjectDetector} instance from {@link ObjectDetectorOptions}.
   *
   * @param modelFile the detection model {@link File} instance
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws AssertionError if error occurs when creating {@link ObjectDetector} from the native
   *     code
   */
  public static ObjectDetector createFromFileAndOptions(
      File modelFile, final ObjectDetectorOptions options) throws IOException {
    try (ParcelFileDescriptor descriptor =
        ParcelFileDescriptor.open(modelFile, ParcelFileDescriptor.MODE_READ_ONLY)) {
      return new ObjectDetector(
          TaskJniUtils.createHandleFromLibrary(
              new TaskJniUtils.EmptyHandleProvider() {
                @Override
                public long createHandle() {
                  return initJniWithModelFdAndOptions(
                      descriptor.getFd(),
                      /*fileDescriptorLength=*/ OPTIONAL_FD_LENGTH,
                      /*fileDescriptorOffset=*/ OPTIONAL_FD_OFFSET,
                      options);
                }
              },
              OBJECT_DETECTOR_NATIVE_LIB));
    }
  }

  /**
   * Creates an {@link ObjectDetector} instance with a model buffer and {@link
   * ObjectDetectorOptions}.
   *
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the detection
   *     model
   * @throws AssertionError if error occurs when creating {@link ObjectDetector} from the native
   *     code
   * @throws IllegalArgumentException if the model buffer is not a direct {@link ByteBuffer} or a
   *     {@link MappedByteBuffer}
   */
  public static ObjectDetector createFromBufferAndOptions(
      final ByteBuffer modelBuffer, final ObjectDetectorOptions options) {
    if (!(modelBuffer.isDirect() || modelBuffer instanceof MappedByteBuffer)) {
      throw new IllegalArgumentException(
          "The model buffer should be either a direct ByteBuffer or a MappedByteBuffer.");
    }
    return new ObjectDetector(
        TaskJniUtils.createHandleFromLibrary(
            new EmptyHandleProvider() {
              @Override
              public long createHandle() {
                return initJniWithByteBuffer(modelBuffer, options);
              }
            },
            OBJECT_DETECTOR_NATIVE_LIB));
  }

  /**
   * Constructor to initialize the JNI with a pointer from C++.
   *
   * @param nativeHandle a pointer referencing memory allocated in C++
   */
  private ObjectDetector(long nativeHandle) {
    super(nativeHandle);
  }

  /** Options for setting up an ObjectDetector. */
  @UsedByReflection("object_detector_jni.cc")
  public static class ObjectDetectorOptions {
    // Not using AutoValue for this class because scoreThreshold cannot have default value
    // (otherwise, the default value would override the one in the model metadata) and `Optional` is
    // not an option here, because
    // 1. java.util.Optional require Java 8 while we need to support Java 7.
    // 2. The Guava library (com.google.common.base.Optional) is avoided in this project. See the
    // comments for labelAllowList.
    private final String displayNamesLocale;
    private final int maxResults;
    private final float scoreThreshold;
    private final boolean isScoreThresholdSet;
    // As an open source project, we've been trying avoiding depending on common java libraries,
    // such as Guava, because it may introduce conflicts with clients who also happen to use those
    // libraries. Therefore, instead of using ImmutableList here, we convert the List into
    // unmodifiableList in setLabelAllowList() and setLabelDenyList() to make it less
    // vulnerable.
    private final List<String> labelAllowList;
    private final List<String> labelDenyList;
    private final int numThreads;

    public static Builder builder() {
      return new Builder();
    }

    /** A builder that helps to configure an instance of ObjectDetectorOptions. */
    public static class Builder {
      private String displayNamesLocale = "en";
      private int maxResults = -1;
      private float scoreThreshold;
      private boolean isScoreThresholdSet = false;
      private List<String> labelAllowList = new ArrayList<>();
      private List<String> labelDenyList = new ArrayList<>();
      private int numThreads = -1;

      private Builder() {}

      /**
       * Sets the locale to use for display names specified through the TFLite Model Metadata, if
       * any.
       *
       * <p>Defaults to English({@code "en"}). See the <a
       * href="https://github.com/tensorflow/tflite-support/blob/3ce83f0cfe2c68fecf83e019f2acc354aaba471f/tensorflow_lite_support/metadata/metadata_schema.fbs#L147">TFLite
       * Metadata schema file.</a> for the accepted pattern of locale.
       */
      public Builder setDisplayNamesLocale(String displayNamesLocale) {
        this.displayNamesLocale = displayNamesLocale;
        return this;
      }

      /**
       * Sets the maximum number of top-scored detection results to return.
       *
       * <p>If < 0, all available results will be returned. If 0, an invalid argument error is
       * returned. Note that models may intrinsically be limited to returning a maximum number of
       * results N: if the provided value here is above N, only N results will be returned. Defaults
       * to -1.
       *
       * @throws IllegalArgumentException if maxResults is 0.
       */
      public Builder setMaxResults(int maxResults) {
        if (maxResults == 0) {
          throw new IllegalArgumentException("maxResults cannot be 0.");
        }
        this.maxResults = maxResults;
        return this;
      }

      /**
       * Sets the score threshold that overrides the one provided in the model metadata (if any).
       * Results below this value are rejected.
       */
      public Builder setScoreThreshold(float scoreThreshold) {
        this.scoreThreshold = scoreThreshold;
        this.isScoreThresholdSet = true;
        return this;
      }

      /**
       * Sets the optional allow list of labels.
       *
       * <p>If non-empty, detection results whose label is not in this set will be filtered out.
       * Duplicate or unknown labels are ignored. Mutually exclusive with {@code labelDenyList}. It
       * will cause {@link AssertionError} when calling {@link #createFromFileAndOptions}, if both
       * {@code labelDenyList} and {@code labelAllowList} are set.
       */
      public Builder setLabelAllowList(List<String> labelAllowList) {
        this.labelAllowList = Collections.unmodifiableList(new ArrayList<>(labelAllowList));
        return this;
      }

      /**
       * Sets the optional deny list of labels.
       *
       * <p>If non-empty, detection results whose label is in this set will be filtered out.
       * Duplicate or unknown labels are ignored. Mutually exclusive with {@code labelAllowList}. It
       * will cause {@link AssertionError} when calling {@link #createFromFileAndOptions}, if both
       * {@code labelDenyList} and {@code labelAllowList} are set.
       */
      public Builder setLabelDenyList(List<String> labelDenyList) {
        this.labelDenyList = Collections.unmodifiableList(new ArrayList<>(labelDenyList));
        return this;
      }

      /**
       * Sets the number of threads to be used for TFLite ops that support multi-threading when
       * running inference with CPU. Defaults to -1.
       *
       * <p>numThreads should be greater than 0 or equal to -1. Setting numThreads to -1 has the
       * effect to let TFLite runtime set the value.
       */
      public Builder setNumThreads(int numThreads) {
        this.numThreads = numThreads;
        return this;
      }

      public ObjectDetectorOptions build() {
        return new ObjectDetectorOptions(this);
      }
    }

    @UsedByReflection("object_detector_jni.cc")
    public String getDisplayNamesLocale() {
      return displayNamesLocale;
    }

    @UsedByReflection("object_detector_jni.cc")
    public int getMaxResults() {
      return maxResults;
    }

    @UsedByReflection("object_detector_jni.cc")
    public float getScoreThreshold() {
      return scoreThreshold;
    }

    @UsedByReflection("object_detector_jni.cc")
    public boolean getIsScoreThresholdSet() {
      return isScoreThresholdSet;
    }

    @UsedByReflection("object_detector_jni.cc")
    public List<String> getLabelAllowList() {
      return new ArrayList<>(labelAllowList);
    }

    @UsedByReflection("object_detector_jni.cc")
    public List<String> getLabelDenyList() {
      return new ArrayList<>(labelDenyList);
    }

    @UsedByReflection("object_detector_jni.cc")
    public int getNumThreads() {
      return numThreads;
    }

    private ObjectDetectorOptions(Builder builder) {
      displayNamesLocale = builder.displayNamesLocale;
      maxResults = builder.maxResults;
      scoreThreshold = builder.scoreThreshold;
      isScoreThresholdSet = builder.isScoreThresholdSet;
      labelAllowList = builder.labelAllowList;
      labelDenyList = builder.labelDenyList;
      numThreads = builder.numThreads;
    }
  }

  /**
   * Performs actual detection on the provided image.
   *
   * <p>{@link ObjectDetector} supports the following {@link TensorImage} color space types:
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
   * @throws AssertionError if error occurs when processing the image from the native code
   * @throws IllegalArgumentException if the color space type of image is unsupported
   */
  public List<Detection> detect(TensorImage image) {
    return detect(image, ImageProcessingOptions.builder().build());
  }

  /**
   * Performs actual detection on the provided image.
   *
   * <p>{@link ObjectDetector} supports the following {@link TensorImage} color space types:
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
   * @param options {@link ObjectDetector} only supports image rotation (through {@link
   *     ImageProcessingOptions.Builder#setOrientation}) currently. The orientation of an image
   *     defaults to {@link ImageProcessingOptions.Orientation#TOP_LEFT}.
   * @throws AssertionError if error occurs when processing the image from the native code
   * @throws IllegalArgumentException if the color space type of image is unsupported
   */
  public List<Detection> detect(TensorImage image, ImageProcessingOptions options) {
    return run(
        new InferenceProvider<List<Detection>>() {
          @Override
          public List<Detection> run(
              long frameBufferHandle, int width, int height, ImageProcessingOptions options) {
            return detect(frameBufferHandle, options);
          }
        },
        image,
        options);
  }

  private List<Detection> detect(long frameBufferHandle, ImageProcessingOptions options) {
    checkNotClosed();

    return detectNative(getNativeHandle(), frameBufferHandle);
  }

  private static native long initJniWithModelFdAndOptions(
      int fileDescriptor,
      long fileDescriptorLength,
      long fileDescriptorOffset,
      ObjectDetectorOptions options);

  private static native long initJniWithByteBuffer(
      ByteBuffer modelBuffer, ObjectDetectorOptions options);

  private static native List<Detection> detectNative(long nativeHandle, long frameBufferHandle);

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
