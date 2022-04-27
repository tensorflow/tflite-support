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
"""Tests for image_searcher."""

import enum

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2
from tensorflow_lite_support.python.task.processor.proto import embedding_options_pb2
from tensorflow_lite_support.python.task.processor.proto import search_options_pb2
from tensorflow_lite_support.python.task.vision import image_searcher
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_pb2.BaseOptions
_EmbeddingOptions = embedding_options_pb2.EmbeddingOptions
_SearchOptions = search_options_pb2.SearchOptions
_ImageSearcher = image_searcher.ImageSearcher
_ImageSearcherOptions = image_searcher.ImageSearcherOptions

_MOBILENET_MODEL = 'mobilenet_v3_small_100_224_embedder.tflite'
_MOBILENET_INDEX = 'searcher_index.ldb'
_EXPECTED_MOBILENET_SEARCH_PARAMS = """
nearest_neighbors { metadata: "burger" distance: -0.0 }
nearest_neighbors { metadata: "car" distance: 1.822435 }
nearest_neighbors { metadata: "bird" distance: 1.930939 }
nearest_neighbors { metadata: "dog" distance: 2.047355 }
nearest_neighbors { metadata: "cat" distance: 2.075868 }
"""

_IMAGE_FILE = 'burger.jpg'


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageSearcherTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image_path = test_util.get_test_data_path(_IMAGE_FILE)
    self.model_path = test_util.get_test_data_path(_MOBILENET_MODEL)
    self.index_path = test_util.get_test_data_path(_MOBILENET_INDEX)

  @parameterized.parameters(
      (_MOBILENET_MODEL, _MOBILENET_INDEX, True, False, ModelFileType.FILE_NAME,
       _EXPECTED_MOBILENET_SEARCH_PARAMS),
      (_MOBILENET_MODEL, _MOBILENET_INDEX, True, False,
       ModelFileType.FILE_CONTENT, _EXPECTED_MOBILENET_SEARCH_PARAMS),
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

    index_path = test_util.get_test_data_path(index_name)
    options = _ImageSearcherOptions(
        base_options,
        _EmbeddingOptions(l2_normalize=l2_normalize, quantize=quantize),
        _SearchOptions(index_file_name=index_path))
    searcher = _ImageSearcher.create_from_options(options)

    # Loads image.
    image = tensor_image.TensorImage.create_from_file(self.test_image_path)

    # Perform image search.
    image_search_result = searcher.search(image)

    # Comparing results.
    self.assertProtoEquals(expected_result_text_proto, image_search_result)

    # Get user info and compare values.
    self.assertEqual(searcher.get_user_info(), 'userinfo')

  def test_search_with_bounding_box(self):
    # Create searcher.
    searcher = _ImageSearcher.create_from_file(self.model_path, self.index_path)

    # Loads image.
    image = tensor_image.TensorImage.create_from_file(self.test_image_path)

    # Bounding box in "burger.jpg" corresponding to "burger_crop.jpg".
    bounding_box = bounding_box_pb2.BoundingBox(
        origin_x=0, origin_y=0, width=400, height=325)

    # Perform image search.
    image_search_result = searcher.search(image, bounding_box)

    # Expected results.
    expected_result_text_proto = """
    nearest_neighbors { metadata: "burger" distance: 0.134547 }
    nearest_neighbors { metadata: "car" distance: 1.819211 }
    nearest_neighbors { metadata: "bird" distance: 1.96461 }
    nearest_neighbors { metadata: "dog" distance: 2.0569 }
    nearest_neighbors { metadata: "cat" distance: 2.062612 }
    """

    # Comparing results.
    self.assertProtoEquals(expected_result_text_proto, image_search_result)

    # Get user info and compare values.
    self.assertEqual(searcher.get_user_info(), 'userinfo')


if __name__ == '__main__':
  tf.test.main()
