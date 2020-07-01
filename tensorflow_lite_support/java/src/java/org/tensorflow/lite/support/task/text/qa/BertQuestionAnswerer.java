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

import android.content.Context;
import java.nio.ByteBuffer;
import java.util.List;
import org.tensorflow.lite.support.task.core.BaseTaskApi;
import org.tensorflow.lite.support.task.core.TaskJniUtils;

/** Task API for BertQA models. */
public class BertQuestionAnswerer extends BaseTaskApi implements QuestionAnswerer {
  private static final String BERT_QUESTION_ANSWERER_NATIVE_LIBNAME = "bert_question_answerer_jni";

  private BertQuestionAnswerer(long nativeHandle) {
    super(nativeHandle);
  }

  /**
   * Creates the API instance with a bert model and vocabulary file.
   *
   * <p>One suitable model is: https://tfhub.dev/tensorflow/lite-model/mobilebert/1/default/1
   *
   * <p>TODO(b/149510792): Actually, create{Bert|Albert}QuestionAnswerer support multiple models
   * beyond BERT or Albert literally. May refactor names to indicate their preprocessing methods and
   * usage.
   *
   * @param context android context
   * @param pathToModel file path to the bert model
   * @param pathToVocab file path to the vocabulary file
   * @return {@link BertQuestionAnswerer} instance
   */
  public static BertQuestionAnswerer createBertQuestionAnswerer(
      Context context, String pathToModel, String pathToVocab) {
    return new BertQuestionAnswerer(
        TaskJniUtils.createHandleWithMultipleAssetFilesFromLibrary(
            context,
            BertQuestionAnswerer::initJniWithBertByteBuffers,
            BERT_QUESTION_ANSWERER_NATIVE_LIBNAME,
            pathToModel,
            pathToVocab));
  }

  /**
   * Creates the API instance with an albert model and sentence piece model file.
   *
   * <p>One suitable model is: https://tfhub.dev/tensorflow/lite-model/albert_lite_base/squadv1/1
   *
   * @param context android context
   * @param pathToModel file path to the albert model
   * @param pathToSentencePieceModel file path to the sentence piece model file
   * @return {@link BertQuestionAnswerer} instance
   */
  public static BertQuestionAnswerer createAlbertQuestionAnswerer(
      Context context, String pathToModel, String pathToSentencePieceModel) {
    return new BertQuestionAnswerer(
        TaskJniUtils.createHandleWithMultipleAssetFilesFromLibrary(
            context,
            BertQuestionAnswerer::initJniWithAlbertByteBuffers,
            BERT_QUESTION_ANSWERER_NATIVE_LIBNAME,
            pathToModel,
            pathToSentencePieceModel));
  }

  @Override
  public List<QaAnswer> answer(String context, String question) {
    return answerNative(getNativeHandle(), context, question);
  }

  // modelBuffers[0] is tflite model file buffer, and modelBuffers[1] is vocab file buffer.
  private static native long initJniWithBertByteBuffers(ByteBuffer... modelBuffers);

  // modelBuffers[0] is tflite model file buffer, and modelBuffers[1] is sentencepiece model file
  // buffer.
  private static native long initJniWithAlbertByteBuffers(ByteBuffer... modelBuffers);

  private static native List<QaAnswer> answerNative(
      long nativeHandle, String context, String question);
}
