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
"""Tests for object detector."""

import enum
import json

from absl.testing import parameterized
from google.protobuf import json_format
# TODO(b/220067158): Change to import tensorflow and leverage tf.test once
# fixed the dependency issue.
import unittest
from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor.proto import class_pb2
from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2
from tensorflow_lite_support.python.task.processor.proto import detection_options_pb2
from tensorflow_lite_support.python.task.processor.proto import detections_pb2
from tensorflow_lite_support.python.task.vision import object_detector
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.test import base_test
from tensorflow_lite_support.python.test import test_util

_BaseOptions = task_options.BaseOptions
_ExternalFile = task_options.ExternalFile
_ObjectDetector = object_detector.ObjectDetector
_ObjectDetectorOptions = object_detector.ObjectDetectorOptions

_MODEL_FLOAT = 'coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.tflite'


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ObjectDetectorTest(parameterized.TestCase, base_test.BaseTestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_MODEL_FLOAT)

  @staticmethod
  def create_detector_from_options(model_file, **detection_options):
    base_options = _BaseOptions(model_file=model_file)
    detection_options = detection_options_pb2.DetectionOptions(
        **detection_options)
    options = _ObjectDetectorOptions(
        base_options=base_options,
        detection_options=detection_options)
    detector = _ObjectDetector.create_from_options(options)
    return detector

  @staticmethod
  def build_test_data(expected_detections):
    expected_result = detections_pb2.DetectionResult()

    for index in range(len(expected_detections)):
      bounding_box, category = expected_detections[index]
      detection = detections_pb2.Detection()
      detection.bounding_box.CopyFrom(
        bounding_box_pb2.BoundingBox(**bounding_box))
      detection.classes.append(
        class_pb2.Category(**category))
      expected_result.detections.append(detection)

    expected_result_dict = json.loads(
      json_format.MessageToJson(expected_result))

    return expected_result_dict

  @parameterized.parameters((ModelFileType.FILE_NAME, 0.5, [
    (
      {
        'origin_x': 54, 'origin_y': 396, 'width': 393, 'height': 196
      },
      {
        'index': 16,
        'score': 0.64453125,
        'class_name': 'cat'
      }
    ), (
      {
        'origin_x': 602, 'origin_y': 157, 'width': 394, 'height': 447
      }, {
        'index': 16,
        'score': 0.59765625,
        'class_name': 'cat'
      }
    ), (
      {
        'origin_x': 261, 'origin_y': 394, 'width': 179, 'height': 209
      }, {
        'index': 16,
        'score': 0.5625,
        'class_name': 'cat'
      }
    ), (
      {
        'origin_x': 389, 'origin_y': 197, 'width': 276, 'height': 409
      }, {
        'index': 17,
        'score': 0.51171875,
        'class_name': 'dog'
      }
    )
  ]), (ModelFileType.FILE_CONTENT, 0.5, [
    (
      {
        'origin_x': 54, 'origin_y': 396, 'width': 393, 'height': 196
      }, {
        'index': 16,
        'score': 0.64453125,
        'class_name': 'cat'
      }
    ), (
      {
        'origin_x': 602, 'origin_y': 157, 'width': 394, 'height': 447
      }, {
        'index': 16,
        'score': 0.59765625,
        'class_name': 'cat'
      }
    ), (
      {
        'origin_x': 261, 'origin_y': 394, 'width': 179, 'height': 209
      }, {
        'index': 16,
        'score': 0.5625,
        'class_name': 'cat'
      }
    ), (
      {
        'origin_x': 389, 'origin_y': 197, 'width': 276, 'height': 409
      }, {
        'index': 17,
        'score': 0.51171875,
        'class_name': 'dog'
      }
    )
  ]))
  def test_detect_model(self, model_file_type, score_threshold, expected_detections):
    # Creates detector.
    if model_file_type is ModelFileType.FILE_NAME:
      model_file = _ExternalFile(file_name=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      model_file = _ExternalFile(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    detector = self.create_detector_from_options(
      model_file=model_file, score_threshold=score_threshold)

    # Loads image.
    image = tensor_image.TensorImage.from_file(
        test_util.get_test_data_path('cats_and_dogs.jpg'))

    # Performs object detection on the input.
    image_result = detector.detect(image)
    image_result_dict = json.loads(json_format.MessageToJson(image_result))

    # Builds test data.
    expected_result_dict = self.build_test_data(expected_detections)

    # Comparing results.
    self.assertDeepAlmostEqual(
        image_result_dict, expected_result_dict, places=5)


if __name__ == '__main__':
  unittest.main()
