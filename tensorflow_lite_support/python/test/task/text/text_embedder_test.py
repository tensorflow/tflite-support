# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for text_embedder."""

import enum

from absl.testing import parameterized

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor.proto import embedding_options_pb2
from tensorflow_lite_support.python.task.text import text_embedder
from tensorflow_lite_support.python.test import test_util
import unittest

_BaseOptions = task_options.BaseOptions
_ExternalFile = task_options.ExternalFile
_TextEmbedder = text_embedder.TextEmbedder
_TextEmbedderOptions = text_embedder.TextEmbedderOptions

_REGEX_MODEL = "regex_one_embedding_with_metadata.tflite"
_BERT_MODEL = "mobilebert_embedding_with_metadata.tflite"


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class TextEmbedderTest(parameterized.TestCase, unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_REGEX_MODEL)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    options = _TextEmbedderOptions(
        _BaseOptions(model_file=_ExternalFile(file_name=self.model_path)))
    embedder = _TextEmbedder.create_from_options(options)
    self.assertIsInstance(embedder, _TextEmbedder)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        Exception,
        r"INVALID_ARGUMENT: Missing mandatory `model_file` field in "
        r"`base_options` \[tflite::support::TfLiteSupportStatus='2'\]"):
      options = _TextEmbedderOptions(
          _BaseOptions(model_file=_ExternalFile(file_name="")))
      _TextEmbedder.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, "rb") as f:
      options = _TextEmbedderOptions(
          _BaseOptions(model_file=_ExternalFile(file_content=f.read())))
      embedder = _TextEmbedder.create_from_options(options)
      self.assertIsInstance(embedder, _TextEmbedder)

  @parameterized.parameters(
      (_REGEX_MODEL, False, False, ModelFileType.FILE_NAME, 16, 0.999937),
      (_REGEX_MODEL, True, True, ModelFileType.FILE_NAME, 16, 0.999878),
      (_BERT_MODEL, False, False, ModelFileType.FILE_CONTENT, 512, 0.969514),
      (_BERT_MODEL, True, True, ModelFileType.FILE_CONTENT, 512, 0.966984),
  )
  def test_embed(self, model_name, l2_normalize, quantize, model_file_type,
                 embedding_length, expected_similarity):
    # Create embedder.
    model_path = test_util.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      model_file = _ExternalFile(file_name=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, "rb") as f:
        model_content = f.read()
      model_file = _ExternalFile(file_content=model_content)
    else:
      # Should never happen
      raise ValueError("model_file_type is invalid.")

    options = _TextEmbedderOptions(
        _BaseOptions(model_file),
        embedding_options_pb2.EmbeddingOptions(
            l2_normalize=l2_normalize, quantize=quantize))
    embedder = _TextEmbedder.create_from_options(options)

    # Extract embeddings.
    result0 = embedder.embed("it's a charming and often affecting journey")
    result1 = embedder.embed("what a great and fantastic trip")

    # Check embedding sizes.
    def _check_embedding_size(result):
      self.assertLen(result.embeddings, 1)
      feature_vector = result.embeddings[0].feature_vector
      if quantize:
        self.assertLen(feature_vector.value_string, embedding_length)
      else:
        self.assertLen(feature_vector.value_float, embedding_length)

    _check_embedding_size(result0)
    _check_embedding_size(result1)

  def test_get_embedding_dimension(self):
    options = _TextEmbedderOptions(
        _BaseOptions(model_file=_ExternalFile(file_name=self.model_path)))
    embedder = _TextEmbedder.create_from_options(options)
    self.assertEqual(embedder.get_embedding_dimension(0), 16)
    self.assertEqual(embedder.get_embedding_dimension(1), -1)

  def test_number_of_output_layers(self):
    options = _TextEmbedderOptions(
        _BaseOptions(model_file=_ExternalFile(file_name=self.model_path)))
    embedder = _TextEmbedder.create_from_options(options)
    self.assertEqual(embedder.number_of_output_layers, 1)


if __name__ == "__main__":
  unittest.main()
