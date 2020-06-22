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

package org.tensorflow.lite.support.task.text.qa;

import androidx.test.core.app.ApplicationProvider;
import java.util.List;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/** Test for {@link BertQuestionAnswerer}. */
@RunWith(RobolectricTestRunner.class)
public class BertQuestionAnswererTest {
  private static final String BERT_MODEL_FILE = "mobile_bert.tflite";
  private static final String VOCAB_FILE = "vocab.txt";
  private static final String ALBERT_MODEL_FILE = "albert.tflite";
  private static final String SP_MODEL_FILE = "30k-clean.model";
  private static final String QUESTION = "what's the color of bunny?";
  private static final String ANSWER = "white";
  private static final String CONTEXT =
      "A bunny is white and it's very fluffy. You don't want to eat a bunny because bunny is so"
          + " cute.";

  @Test
  public void verifyBertAnswer() {
    BertQuestionAnswerer bertQuestionAnswerer =
        BertQuestionAnswerer.createBertQuestionAnswerer(
            ApplicationProvider.getApplicationContext(), BERT_MODEL_FILE, VOCAB_FILE);
    List<QaAnswer> answers = bertQuestionAnswerer.answer(CONTEXT, QUESTION);
    Assert.assertEquals(ANSWER, answers.get(0).text);
  }

  @Test
  public void verifyAlbertAnswer() {
    BertQuestionAnswerer albertQuestionAnswerer =
        BertQuestionAnswerer.createAlbertQuestionAnswerer(
            ApplicationProvider.getApplicationContext(), ALBERT_MODEL_FILE, SP_MODEL_FILE);
    List<QaAnswer> answers = albertQuestionAnswerer.answer(CONTEXT, QUESTION);
    Assert.assertEquals(ANSWER, answers.get(0).text);
  }
}
