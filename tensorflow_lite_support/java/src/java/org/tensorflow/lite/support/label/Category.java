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

package org.tensorflow.lite.support.label;

import com.google.auto.value.AutoValue;
import org.tensorflow.lite.annotations.UsedByReflection;

/**
 * Category is a util class, contains a label and a float value. Typically it's used as result of
 * classification tasks.
 */
@AutoValue
@UsedByReflection("ClassifierJNI")
public abstract class Category {

  @UsedByReflection("ClassifierJNI")
  public static Category create(String label, float score) {
    return new AutoValue_Category(label, score);
  }

  public abstract String getLabel();

  public abstract float getScore();
}
