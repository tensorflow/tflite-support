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
from tensorflow_lite_support.python.task.processor.proto import search_result_pb2
from tensorflow_lite_support.python.task.text import text_searcher
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_pb2.BaseOptions
_EmbeddingOptions = embedding_options_pb2.EmbeddingOptions
_SearchOptions = search_options_pb2.SearchOptions
_TextSearcher = text_searcher.TextSearcher
_TextSearcherOptions = text_searcher.TextSearcherOptions

_REGEX_MODEL = 'regex_one_embedding_with_metadata.tflite'
_REGEX_INDEX = 'regex_index.ldb'
_EXPECTED_REGEX_SEARCH_PARAMS = [
  {
    'metadata': 'The weather was excellent.',
    'distance': 0.0
  }, {
    'metadata': 'The sun was shining on that day.',
    'distance': 5.7e-5
  }, {
    'metadata': 'The cat is chasing after the mouse.',
    'distance': 8.9e-5
  }, {
    'metadata': 'It was a sunny day.',
    'distance': 0.000113
  }, {
    'metadata': 'He was very happy with his newly bought car.',
    'distance': 0.000119
  }
]


def _build_test_data(expected_nearest_neighbors):
  expected_search_result = search_result_pb2.SearchResult()
  expected_search_result.nearest_neighbors.extend(
    [search_result_pb2.NearestNeighbor(
      metadata=nearest_neighbor['metadata'].encode(),
      distance=nearest_neighbor['distance'])
     for nearest_neighbor in expected_nearest_neighbors])

  return expected_search_result


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class TextSearcherTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_REGEX_MODEL)
    self.index_path = test_util.get_test_data_path(_REGEX_INDEX)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    index_file = external_file_pb2.ExternalFile(file_name=self.index_path)
    options = _TextSearcherOptions(
      base_options=_BaseOptions(file_name=self.model_path),
      search_options=_SearchOptions(index_file=index_file))
    searcher = _TextSearcher.create_from_options(options)
    self.assertIsInstance(searcher, _TextSearcher)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, "rb") as f:
      index_file = external_file_pb2.ExternalFile(file_name=self.index_path)
      options = _TextSearcherOptions(
        base_options=_BaseOptions(file_content=f.read()),
        search_options=_SearchOptions(index_file=index_file))
      searcher = _TextSearcher.create_from_options(options)
      self.assertIsInstance(searcher, _TextSearcher)

  def test_create_from_options_fails_with_invalid_index_path(self):
    # Invalid index path.
    with self.assertRaisesRegex(
        ValueError,
        r"Missing mandatory `index_file` field in `search_options`"):
      options = _TextSearcherOptions(
        base_options=_BaseOptions(file_name=self.model_path))
      _TextSearcher.create_from_options(options)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name' or 'file_descriptor_meta'."):
      index_file = external_file_pb2.ExternalFile(file_name=self.index_path)
      options = _TextSearcherOptions(
        base_options=_BaseOptions(file_name=""),
        search_options=_SearchOptions(index_file=index_file))
      _TextSearcher.create_from_options(options)

  def test_create_from_options_fails_with_invalid_quantization(self):
    # Invalid quantization option.
    with self.assertRaisesRegex(
        ValueError,
        r"Setting EmbeddingOptions.normalize = true is not allowed in "
        r"searchers."):
      index_file = external_file_pb2.ExternalFile(file_name=self.index_path)
      options = _TextSearcherOptions(
        base_options=_BaseOptions(file_name=self.model_path),
        embedding_options=_EmbeddingOptions(quantize=True),
        search_options=_SearchOptions(index_file=index_file))
      _TextSearcher.create_from_options(options)

  @parameterized.parameters(
    (_REGEX_MODEL, _REGEX_INDEX, True, False, ModelFileType.FILE_NAME,
     _EXPECTED_REGEX_SEARCH_PARAMS),
  )
  def test_search(self, model_name, index_name, l2_normalize, quantize,
                  model_file_type, expected_search_params):
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

    index_file_name = test_util.get_test_data_path(index_name)
    index_file = external_file_pb2.ExternalFile(file_name=index_file_name)
    options = _TextSearcherOptions(
      base_options,
      _EmbeddingOptions(l2_normalize=l2_normalize, quantize=quantize),
      _SearchOptions(index_file=index_file))
    searcher = _TextSearcher.create_from_options(options)

    # Perform text search.
    text_search_result = searcher.search("The weather was excellent.")

    # Build test data.
    expected_search_result = _build_test_data(expected_search_params)

    # Check nearest-neighbour sizes for the actual and expected results.
    self.assertEqual(len(text_search_result.nearest_neighbors),
                     len(expected_search_result.nearest_neighbors))

    actual_search_result = search_result_pb2.SearchResult()
    actual_search_result.ParseFromString(text_search_result.SerializeToString())
    self.assertProtoEquals(actual_search_result, expected_search_result)


if __name__ == "__main__":
  tf.test.main()
