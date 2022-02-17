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
"""Tests for image_classifier."""

import enum
import json

from absl.testing import parameterized
from google.protobuf import json_format
# TODO(b/220067158): Change to import tensorflow and leverage tf.test once
# fixed the dependency issue.
import unittest
from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor.proto import class_pb2
from tensorflow_lite_support.python.task.processor.proto import classification_options_pb2
from tensorflow_lite_support.python.task.processor.proto import classifications_pb2
from tensorflow_lite_support.python.task.vision import image_classifier
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.test import base_test
from tensorflow_lite_support.python.test import test_util

_BaseOptions = task_options.BaseOptions
_ExternalFile = task_options.ExternalFile
_ImageClassifier = image_classifier.ImageClassifier
_ImageClassifierOptions = image_classifier.ImageClassifierOptions

_MODEL_FLOAT = 'mobilenet_v2_1.0_224.tflite'


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageClassifierTest(parameterized.TestCase, base_test.BaseTestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_MODEL_FLOAT)

  @staticmethod
  def create_classifier_from_options(model_file, **classification_options):
    base_options = _BaseOptions(model_file=model_file)
    classification_options = classification_options_pb2.ClassificationOptions(
        **classification_options)
    options = _ImageClassifierOptions(
        base_options=base_options,
        classification_options=classification_options)
    classifier = _ImageClassifier.create_from_options(options)
    return classifier

  @staticmethod
  def build_test_data(expected_categories):
    classifications = classifications_pb2.Classifications(head_index=0)
    classifications.classes.extend(
        [class_pb2.Category(**args) for args in expected_categories])
    expected_result = classifications_pb2.ClassificationResult()
    expected_result.classifications.append(classifications)
    expected_result_dict = json.loads(
        json_format.MessageToJson(expected_result))

    return expected_result_dict

  @parameterized.parameters((ModelFileType.FILE_NAME, 3, [{
      'index': 934,
      'score': 0.7399742007255554,
      'class_name': 'cheeseburger'
  }, {
      'index': 925,
      'score': 0.026928534731268883,
      'class_name': 'guacamole'
  }, {
      'index': 932,
      'score': 0.025737214833498,
      'class_name': 'bagel'
  }]), (ModelFileType.FILE_CONTENT, 3, [{
      'index': 934,
      'score': 0.7399742007255554,
      'class_name': 'cheeseburger'
  }, {
      'index': 925,
      'score': 0.026928534731268883,
      'class_name': 'guacamole'
  }, {
      'index': 932,
      'score': 0.025737214833498,
      'class_name': 'bagel'
  }]))
  def test_classify_model(self, model_file_type, max_results,
                          expected_categories):
    # Creates classifier.
    if model_file_type is ModelFileType.FILE_NAME:
      model_file = _ExternalFile(file_name=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      model_file = _ExternalFile(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    classifier = self.create_classifier_from_options(
        model_file, max_results=max_results)

    # Loads image.
    image = tensor_image.TensorImage.from_file(
        test_util.get_test_data_path('burger.jpg'))

    # Classifies the input.
    image_result = classifier.classify(image, bounding_box=None)
    image_result_dict = json.loads(json_format.MessageToJson(image_result))

    # Builds test data.
    expected_result_dict = self.build_test_data(expected_categories)

    # Comparing results (classification w/o bounding box).
    self.assertDeepAlmostEqual(
        image_result_dict, expected_result_dict, places=5)


if __name__ == '__main__':
  unittest.main()
