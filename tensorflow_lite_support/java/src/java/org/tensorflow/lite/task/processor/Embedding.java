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

/** Represents the embed result of a Embedder model. */
@AutoValue
@UsedByReflection("embedder_jni.cc")
public abstract class Embedding {

  @UsedByReflection("embedder_jni.cc")
  static Embedding create(FeatureVector featureVectorArray, int outputIndex) {
    // Ideally, the API should accept a ByteBuffer instead of a byte[]. However, converting byte[]
    // to ByteBuffer in JNI will lead to unnecessarily complex code which involves 6 more reflection
    // calls. We can make this method package private, because users in general shouldn't need to
    // create Embedding instances, but only consume the objects return from Task Library. This
    // API will be used mostly for internal purpose.
    return new AutoValue_Embedding(featureVectorArray, outputIndex);
  }

  /**
   * Gets the user-defined feature vector about the result.
   *
   * <p><b>Do not mutate</b> the returned featureVector.
   */
  public abstract FeatureVector getFeatureVector();

  /** Gets the output index indicating the index of the model output layer that produced this feature vector. */
  public abstract int getOutputIndex();
}
