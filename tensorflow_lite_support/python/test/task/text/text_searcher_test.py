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
from tensorflow_lite_support.python.task.processor.proto import embedding_options_pb2
from tensorflow_lite_support.python.task.processor.proto import search_options_pb2
from tensorflow_lite_support.python.task.text import text_searcher
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_pb2.BaseOptions
_EmbeddingOptions = embedding_options_pb2.EmbeddingOptions
_SearchOptions = search_options_pb2.SearchOptions
_TextSearcher = text_searcher.TextSearcher
_TextSearcherOptions = text_searcher.TextSearcherOptions

_REGEX_MODEL = 'regex_one_embedding_with_metadata.tflite'
_REGEX_INDEX = 'regex_index.ldb'
_EXPECTED_REGEX_SEARCH_PARAMS = """
nearest_neighbors {
  metadata: "The weather was excellent."
  distance: 0.0
}
nearest_neighbors {
  metadata: "The sun was shining on that day."
  distance: 5.7e-5
}
nearest_neighbors {
  metadata: "The cat is chasing after the mouse."
  distance: 8.9e-5
}
nearest_neighbors {
  metadata: "It was a sunny day."
  distance: 0.000113
}
nearest_neighbors {
  metadata: "He was very happy with his newly bought car."
  distance: 0.000119
}
"""


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class TextSearcherTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
    (_REGEX_MODEL, _REGEX_INDEX, True, False, ModelFileType.FILE_NAME,
     _EXPECTED_REGEX_SEARCH_PARAMS),
    (_REGEX_MODEL, _REGEX_INDEX, True, False, ModelFileType.FILE_CONTENT,
     _EXPECTED_REGEX_SEARCH_PARAMS),
  )
  def test_search(self, model_name, index_name, l2_normalize, quantize,
                  model_file_type, expected_result_text_proto):
    # Create searcher.
    model_path = test_util.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    index_file_name = test_util.get_test_data_path(index_name)
    options = _TextSearcherOptions(
      base_options,
      _EmbeddingOptions(l2_normalize=l2_normalize, quantize=quantize),
      _SearchOptions(index_file_name=index_file_name))
    searcher = _TextSearcher.create_from_options(options)

    # Perform text search.
    text_search_result = searcher.search('The weather was excellent.')

    # Comparing results.
    self.assertProtoEquals(expected_result_text_proto, text_search_result)

    # Get user info and compare values.
    self.assertEqual(searcher.get_user_info(), 'userinfo')


if __name__ == "__main__":
  tf.test.main()
