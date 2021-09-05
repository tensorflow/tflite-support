# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for ObjectDetector.MetadataWriter."""

import os

from absl.testing import parameterized

import tensorflow as tf

from tensorflow_lite_support.metadata.python.metadata_writers import object_detector
from tensorflow_lite_support.metadata.python.tests.metadata_writers import test_utils

_PATH = '../testdata/object_detector/'
_LABEL_FILE = '../testdata/object_detector/labelmap.txt'
_NORM_MEAN = 127.5
_NORM_STD = 127.5


class MetadataWriterTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('ssd_mobilenet_v1'),
      ('efficientdet_lite0_v1'),
  )
  def test_create_for_inference_should_succeed(self, model_name: str):
    model_path = os.path.join(_PATH, model_name + '.tflite')
    writer = object_detector.MetadataWriter.create_for_inference(
        test_utils.load_file(model_path), [_NORM_MEAN], [_NORM_STD],
        [_LABEL_FILE])

    json_path = os.path.join(_PATH, model_name + '.json')
    metadata_json = writer.get_metadata_json()
    expected_json = test_utils.load_file(json_path, 'r')
    self.assertEqual(metadata_json, expected_json)

  @parameterized.parameters(
      ('ssd_mobilenet_v1'),
      ('efficientdet_lite0_v1'),
  )
  def test_create_from_metadata_info_by_default_should_succeed(
      self, model_name: str):
    model_path = os.path.join(_PATH, model_name + '.tflite')
    writer = object_detector.MetadataWriter.create_from_metadata_info(
        test_utils.load_file(model_path))

    json_path = os.path.join(_PATH, model_name + '_default.json')
    metadata_json = writer.get_metadata_json()
    expected_json = test_utils.load_file(json_path, 'r')
    self.assertEqual(metadata_json, expected_json)


if __name__ == '__main__':
  tf.test.main()
