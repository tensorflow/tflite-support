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

package org.tensorflow.lite.task.processor;

import com.google.auto.value.AutoValue;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.tensorflow.lite.task.core.annotations.UsedByReflection;

/** Represents the feature vector of a Embedder model. */
@AutoValue
@UsedByReflection("embedder_jni.cc")
public abstract class FeatureVector {

  @UsedByReflection("embedder_jni.cc")
  static FeatureVector create(byte[] valueStringArray, float valueFloat) {
    // Convert byte[] valueStringArray to ByteBuffer which handles endianess better.
    //
    // Ideally, the API should accept a ByteBuffer instead of a byte[]. However, converting byte[]
    // to ByteBuffer in JNI will lead to unnecessarily complex code which involves 6 more reflection
    // calls. We can make this method package private, because users in general shouldn't need to
    // create FeatureVector instances, but only consume the objects return from Task Library. This
    // API will be used mostly for internal purpose.
    ByteBuffer valueString = ByteBuffer.wrap(valueStringArray);
    valueString.order(ByteOrder.nativeOrder());
    return new AutoValue_FeatureVector(valueString, valueFloat);
  }

  /**
   * Gets the scalar-quantized embedding. Only provided if `quantize` is set to true in
   * the ImageEmbedderOptions.
   *
   * <p><b>Do not mutate</b> the returned value string.
   */
  public abstract ByteBuffer getValueString();

  /** 
   * GRaw output of the embedding layer. Only provided if `quantize` is set to
   * false in the EmbeddingOptions, which is the case by default. 
   */
  public abstract float getValueFloat();
}
