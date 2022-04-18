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
"""Tests for text_searcher."""

import enum

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.core.proto import external_file_pb2
from tensorflow_lite_support.python.task.processor.proto import embedding_options_pb2
from tensorflow_lite_support.python.task.processor.proto import search_options_pb2
from tensorflow_lite_support.python.task.text import text_searcher
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_pb2.BaseOptions
_TextSearcher = text_searcher.TextSearcher
_TextSearcherOptions = text_searcher.TextSearcherOptions

_REGEX_MODEL = "regex_one_embedding_with_metadata.tflite"
_BERT_MODEL = "mobilebert_embedding_with_metadata.tflite"


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class TextEmbedderTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_REGEX_MODEL)

  @parameterized.parameters(
      (_REGEX_MODEL, False, False, ModelFileType.FILE_NAME),
  )
  def test_search(self, model_name, l2_normalize, quantize, model_file_type):
    # Create embedder.
    model_path = test_util.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, "rb") as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError("model_file_type is invalid.")

    index_file = external_file_pb2.ExternalFile(
      file_name='/path/to/index_file.ldb')
    options = _TextSearcherOptions(
        base_options,
        embedding_options_pb2.EmbeddingOptions(
            l2_normalize=l2_normalize, quantize=quantize),
       search_options_pb2.SearchOptions(
            index_file=index_file))
    searcher = _TextSearcher.create_from_options(options)

    # Perform text search.
    result = searcher.search("testing")


if __name__ == "__main__":
  tf.test.main()
