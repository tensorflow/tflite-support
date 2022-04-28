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

_BERT_MODEL = 'mobilebert_embedding_with_metadata.tflite'
_BERT_INDEX = 'mobilebert_index.ldb'
_EXPECTED_BERT_SEARCH_PARAMS = """
nearest_neighbors {
  metadata: "The weather was excellent."
  distance: 0.0
}
nearest_neighbors {
  metadata: "It was a sunny day."
  distance: 0.115369
}
nearest_neighbors {
  metadata: "The sun was shining on that day."
  distance: 0.230017
}
nearest_neighbors {
  metadata: "He was very happy with his newly bought car."
  distance: 0.324563
}
nearest_neighbors {
  metadata: "The cat is chasing after the mouse."
  distance: 0.966928
}
"""

_USE_MODEL = 'universal_sentence_encoder_qa_with_metadata.tflite'
_USE_INDEX = 'universal_sentence_encoder_index.ldb'
_EXPECTED_USE_SEARCH_PARAMS = """
nearest_neighbors {
  metadata: "The weather was excellent."
  distance: 0.0
}
nearest_neighbors {
  metadata: "It was a sunny day."
  distance: 0.146359
}
nearest_neighbors {
  metadata: "The sun was shining on that day."
  distance: 0.152225
}
nearest_neighbors {
  metadata: "The cat is chasing after the mouse."
  distance: 0.359965
}
nearest_neighbors {
  metadata: "He was very happy with his newly bought car."
  distance: 0.366927
}
"""

_MAX_RESULTS = 2


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class IndexFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class TextSearcherTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_REGEX_MODEL)
    self.index_path = test_util.get_test_data_path(_REGEX_INDEX)

  def test_create_from_file_succeeds_with_valid_model_and_index_paths(self):
    # Creates with default option and valid model and index files successfully.
    searcher = _TextSearcher.create_from_file(self.model_path, self.index_path)
    self.assertIsInstance(searcher, _TextSearcher)

  def test_create_from_options_succeeds_with_valid_model_and_index_paths(self):
    options = _TextSearcherOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        search_options=_SearchOptions(index_file_name=self.index_path))
    searcher = _TextSearcher.create_from_options(options)
    self.assertIsInstance(searcher, _TextSearcher)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      options = _TextSearcherOptions(
          base_options=_BaseOptions(file_content=f.read()),
          search_options=_SearchOptions(index_file_name=self.index_path))
      searcher = _TextSearcher.create_from_options(options)
      self.assertIsInstance(searcher, _TextSearcher)

  def test_create_from_options_succeeds_with_valid_index_content(self):
    # Creates with options containing index content successfully.
    with open(self.index_path, 'rb') as f:
      options = _TextSearcherOptions(
          base_options=_BaseOptions(file_name=self.model_path),
          search_options=_SearchOptions(index_file_content=f.read()))
      searcher = _TextSearcher.create_from_options(options)
      self.assertIsInstance(searcher, _TextSearcher)

  def test_create_from_options_fails_with_invalid_index_path(self):
    # Invalid index path.
    with self.assertRaisesRegex(
        ValueError,
        r'Missing mandatory `index_file` field in `search_options`'):
      options = _TextSearcherOptions(
          base_options=_BaseOptions(file_name=self.model_path))
      _TextSearcher.create_from_options(options)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name' or 'file_descriptor_meta'."):
      options = _TextSearcherOptions(
          base_options=_BaseOptions(file_name=''),
          search_options=_SearchOptions(index_file_name=self.index_path))
      _TextSearcher.create_from_options(options)

  def test_create_from_options_fails_with_invalid_quantization(self):
    # Invalid quantization option.
    with self.assertRaisesRegex(
        ValueError,
        r'Setting EmbeddingOptions.quantize = true is not allowed in '
        r'searchers.'):
      options = _TextSearcherOptions(
          base_options=_BaseOptions(file_name=self.model_path),
          embedding_options=_EmbeddingOptions(quantize=True),
          search_options=_SearchOptions(index_file_name=self.index_path))
      _TextSearcher.create_from_options(options)

  def test_create_from_options_fails_with_invalid_max_results(self):
    # Invalid max results option.
    with self.assertRaisesRegex(
        ValueError, r'SearchOptions.max_results must be > 0, found -1.'):
      options = _TextSearcherOptions(
          base_options=_BaseOptions(file_name=self.model_path),
          search_options=_SearchOptions(
              index_file_name=self.index_path, max_results=-1))
      _TextSearcher.create_from_options(options)

  @parameterized.parameters(
      (_REGEX_MODEL, _REGEX_INDEX, True, False, ModelFileType.FILE_NAME,
       IndexFileType.FILE_NAME, _EXPECTED_REGEX_SEARCH_PARAMS),
      (_REGEX_MODEL, _REGEX_INDEX, True, False, ModelFileType.FILE_CONTENT,
       IndexFileType.FILE_NAME, _EXPECTED_REGEX_SEARCH_PARAMS),
      (_REGEX_MODEL, _REGEX_INDEX, True, False, ModelFileType.FILE_NAME,
       IndexFileType.FILE_CONTENT, _EXPECTED_REGEX_SEARCH_PARAMS),
      (_REGEX_MODEL, _REGEX_INDEX, True, False, ModelFileType.FILE_CONTENT,
       IndexFileType.FILE_CONTENT, _EXPECTED_REGEX_SEARCH_PARAMS),
      (_BERT_MODEL, _BERT_INDEX, True, False, ModelFileType.FILE_NAME,
       IndexFileType.FILE_NAME, _EXPECTED_BERT_SEARCH_PARAMS),
      (_BERT_MODEL, _BERT_INDEX, True, False, ModelFileType.FILE_CONTENT,
       IndexFileType.FILE_NAME, _EXPECTED_BERT_SEARCH_PARAMS),
      (_BERT_MODEL, _BERT_INDEX, True, False, ModelFileType.FILE_NAME,
       IndexFileType.FILE_CONTENT, _EXPECTED_BERT_SEARCH_PARAMS),
      (_BERT_MODEL, _BERT_INDEX, True, False, ModelFileType.FILE_CONTENT,
       IndexFileType.FILE_CONTENT, _EXPECTED_BERT_SEARCH_PARAMS),
      (_USE_MODEL, _USE_INDEX, True, False, ModelFileType.FILE_NAME,
       IndexFileType.FILE_NAME, _EXPECTED_USE_SEARCH_PARAMS),
      (_USE_MODEL, _USE_INDEX, True, False, ModelFileType.FILE_CONTENT,
       IndexFileType.FILE_NAME, _EXPECTED_USE_SEARCH_PARAMS),
      (_USE_MODEL, _USE_INDEX, True, False, ModelFileType.FILE_NAME,
       IndexFileType.FILE_CONTENT, _EXPECTED_USE_SEARCH_PARAMS),
      (_USE_MODEL, _USE_INDEX, True, False, ModelFileType.FILE_CONTENT,
       IndexFileType.FILE_CONTENT, _EXPECTED_USE_SEARCH_PARAMS))
  def test_search(self, model_name, index_name, l2_normalize, quantize,
                  model_file_type, index_file_type, expected_result_text_proto):
    # Create BaseOptions.
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

    # Create SearchOptions.
    index_path = test_util.get_test_data_path(index_name)
    if index_file_type is IndexFileType.FILE_NAME:
      search_options = _SearchOptions(index_file_name=index_path)
    elif index_file_type is IndexFileType.FILE_CONTENT:
      with open(index_path, 'rb') as f:
        index_content = f.read()
      search_options = _SearchOptions(index_file_content=index_content)
    else:
      # Should never happen
      raise ValueError('index_file_type is invalid.')

    # Create searcher.
    options = _TextSearcherOptions(
        base_options,
        _EmbeddingOptions(l2_normalize=l2_normalize, quantize=quantize),
        search_options)
    searcher = _TextSearcher.create_from_options(options)

    # Perform text search.
    text_search_result = searcher.search('The weather was excellent.')

    # Comparing results.
    self.assertProtoEquals(expected_result_text_proto, text_search_result)

    # Get user info and compare values.
    self.assertEqual(searcher.get_user_info(), 'userinfo')

  def test_max_results_option(self):
    # Create searcher.
    base_options = _BaseOptions(file_name=self.model_path)
    search_options = _SearchOptions(
        index_file_name=self.index_path, max_results=_MAX_RESULTS)
    options = _TextSearcherOptions(base_options,
                                   _EmbeddingOptions(l2_normalize=True),
                                   search_options)
    searcher = _TextSearcher.create_from_options(options)

    # Perform text search.
    text_search_result = searcher.search('The weather was excellent.')
    nearest_neighbors = text_search_result.nearest_neighbors

    self.assertLessEqual(
        len(nearest_neighbors), _MAX_RESULTS, 'Too many results returned.')


if __name__ == '__main__':
  tf.test.main()
