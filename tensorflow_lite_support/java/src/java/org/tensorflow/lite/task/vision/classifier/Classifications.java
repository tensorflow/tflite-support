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

package org.tensorflow.lite.task.vision.classifier;

import com.google.auto.value.AutoValue;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.tensorflow.lite.annotations.UsedByReflection;
import org.tensorflow.lite.support.label.Category;

// TODO(b/161175702): introduce the multihead model and explain what is a `head`.
// TODO(b/161379260): update Category to show both class name and display name.
/** The classification results of one head in a multihead {@link ImageClassifier}. */
@AutoValue
@UsedByReflection("image_classifier_jni.cc")
public abstract class Classifications {

  @UsedByReflection("image_classifier_jni.cc")
  public static Classifications create(List<Category> categories, int headIndex) {
    return new AutoValue_Classifications(
        Collections.unmodifiableList(new ArrayList<Category>(categories)), headIndex);
  }

  public abstract List<Category> getCategories();

  public abstract int getHeadIndex();
}
