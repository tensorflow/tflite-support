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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.tensorflow.lite.annotations.UsedByReflection;
import org.tensorflow.lite.task.core.BaseTaskApi;
import org.tensorflow.lite.task.core.TaskJniUtils;
import org.tensorflow.lite.task.core.TaskJniUtils.EmptyHandleProvider;
import org.tensorflow.lite.task.core.TaskJniUtils.ModelPathAndOptionsHandleProvider;

/**
 * Performs classification on {@link TensorAudio}.
 *
 * <p>IMPORTANT: This class is under developerment and the API is subjected to changes.
 */
public final class AudioClassifier extends BaseTaskApi {

  private static final String AUDIO_CLASSIFIER_NATIVE_LIB = "task_audio_jni";

  /**
   * Constructor to initialize the JNI with a pointer from C++.
   *
   * @param nativeHandle a pointer referencing memory allocated in C++
   */
  private AudioClassifier(long nativeHandle) {
    super(nativeHandle);
  }

  /**
   * Creates an {@link AudioClassifier} instance from {@link AudioClassifierOptions} and {@link
   * AudioClassifierOptions}.
   *
   * @param modelPath path of the classification model with metadata in the assets
   * @param options an instance of AudioClassifierOptions
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws AssertionError if error occurs when creating {@link AudioClassifier} from the native
   *     code
   */
  public static AudioClassifier createFromFileAndOptions(
      String modelPath, AudioClassifierOptions options) throws IOException {
    return new AudioClassifier(
        TaskJniUtils.createHandleFromModelPathAndOptions(
            new ModelPathAndOptionsHandleProvider<AudioClassifierOptions>() {
              @Override
              public long createHandle(String modelPath, AudioClassifierOptions options) {
                return initJniWithModelPathAndOptions(modelPath, options);
              }
            },
            AUDIO_CLASSIFIER_NATIVE_LIB,
            modelPath,
            options));
  }

  /**
   * Creates an {@link AudioClassifier} instance from {@link AudioClassifierOptions}.
   *
   * @param modelPath path of the classification model with metadata in the assets
   * @throws IOException if an I/O error occurs when loading the tflite model
   * @throws AssertionError if error occurs when creating {@link AudioClassifier} from the native
   *     code
   */
  public static AudioClassifier createFromFile(String modelPath) throws IOException {
    return createFromFileAndOptions(modelPath, AudioClassifierOptions.builder().build());
  }

  /**
   * Creates an {@link AudioClassifier} instance with a model buffer and {@link
   * AudioClassifierOptions}.
   *
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the
   *     classification model
   * @param options an instance of AudioClassifierOptions
   * @throws AssertionError if error occurs when creating {@link AudioClassifier} from the native
   *     code
   * @throws IllegalArgumentException if the model buffer is not a direct {@link ByteBuffer} or a
   *     {@link MappedByteBuffer}
   */
  public static AudioClassifier createFromBufferAndOptions(
      final ByteBuffer modelBuffer, final AudioClassifierOptions options) {
    if (!(modelBuffer.isDirect() || modelBuffer instanceof MappedByteBuffer)) {
      throw new IllegalArgumentException(
          "The model buffer should be either a direct ByteBuffer or a MappedByteBuffer.");
    }
    return new AudioClassifier(
        TaskJniUtils.createHandleFromLibrary(
            new EmptyHandleProvider() {
              @Override
              public long createHandle() {
                return initJniWithByteBufferAndOptions(modelBuffer, options);
              }
            },
            AUDIO_CLASSIFIER_NATIVE_LIB));
  }

  /**
   * Creates an {@link AudioClassifier} instance with a model buffer and the default {@link
   * AudioClassifierOptions}.
   *
   * @param modelBuffer a direct {@link ByteBuffer} or a {@link MappedByteBuffer} of the
   *     classification model
   * @throws AssertionError if error occurs when creating {@link AudioClassifier} from the native
   *     code
   * @throws IllegalArgumentException if the model buffer is not a direct {@link ByteBuffer} or a
   *     {@link MappedByteBuffer}
   */
  public static AudioClassifier createFromBuffer(final ByteBuffer modelBuffer) {
    return createFromBufferAndOptions(modelBuffer, AudioClassifierOptions.builder().build());
  }

  private static native long initJniWithByteBufferAndOptions(
      ByteBuffer modelBuffer, AudioClassifierOptions options);

  private static native long initJniWithModelPathAndOptions(
      String modelPath, AudioClassifierOptions options);

  /** Options for setting up an ImageClassifier. */
  @UsedByReflection("audio_classifier_jni.cc")
  public static class AudioClassifierOptions {
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

    public static Builder builder() {
      return new Builder();
    }

    /** A builder that helps to configure an instance of ImageClassifierOptions. */
    public static class Builder {
      private String displayNamesLocale = "en";
      private int maxResults = -1;
      private float scoreThreshold;
      private boolean isScoreThresholdSet = false;
      private List<String> labelAllowList = new ArrayList<>();
      private List<String> labelDenyList = new ArrayList<>();

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
       * Sets the maximum number of top scored results to return.
       *
       * <p>If < 0, all results will be returned. If 0, an invalid argument error is returned.
       * Defaults to -1.
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
       * Sets the score threshold in [0,1).
       *
       * <p>It overrides the one provided in the model metadata (if any). Results below this value
       * are rejected.
       */
      public Builder setScoreThreshold(float scoreThreshold) {
        this.scoreThreshold = scoreThreshold;
        isScoreThresholdSet = true;
        return this;
      }

      /**
       * Sets the optional allowlist of labels.
       *
       * <p>If non-empty, classifications whose label is not in this set will be filtered out.
       * Duplicate or unknown labels are ignored. Mutually exclusive with labelDenyList.
       */
      public Builder setLabelAllowList(List<String> labelAllowList) {
        this.labelAllowList = Collections.unmodifiableList(new ArrayList<>(labelAllowList));
        return this;
      }

      /**
       * Sets the optional denylist of labels.
       *
       * <p>If non-empty, classifications whose label is in this set will be filtered out. Duplicate
       * or unknown labels are ignored. Mutually exclusive with labelAllowList.
       */
      public Builder setLabelDenyList(List<String> labelDenyList) {
        this.labelDenyList = Collections.unmodifiableList(new ArrayList<>(labelDenyList));
        return this;
      }

      public AudioClassifierOptions build() {
        return new AudioClassifierOptions(this);
      }
    }

    @UsedByReflection("audio_classifier_jni.cc")
    public String getDisplayNamesLocale() {
      return displayNamesLocale;
    }

    @UsedByReflection("audio_classifier_jni.cc")
    public int getMaxResults() {
      return maxResults;
    }

    @UsedByReflection("audio_classifier_jni.cc")
    public float getScoreThreshold() {
      return scoreThreshold;
    }

    @UsedByReflection("audio_classifier_jni.cc")
    public boolean getIsScoreThresholdSet() {
      return isScoreThresholdSet;
    }

    @UsedByReflection("audio_classifier_jni.cc")
    public List<String> getLabelAllowList() {
      return new ArrayList<>(labelAllowList);
    }

    @UsedByReflection("audio_classifier_jni.cc")
    public List<String> getLabelDenyList() {
      return new ArrayList<>(labelDenyList);
    }

    private AudioClassifierOptions(Builder builder) {
      displayNamesLocale = builder.displayNamesLocale;
      maxResults = builder.maxResults;
      scoreThreshold = builder.scoreThreshold;
      isScoreThresholdSet = builder.isScoreThresholdSet;
      labelAllowList = builder.labelAllowList;
      labelDenyList = builder.labelDenyList;
    }
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
