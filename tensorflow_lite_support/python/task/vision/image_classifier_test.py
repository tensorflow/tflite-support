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

from absl.testing import parameterized
from google.protobuf import json_format

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor import processor_options
from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2
from tensorflow_lite_support.python.task.processor.proto import classifications_pb2
from tensorflow_lite_support.python.task.processor.proto import class_pb2
from tensorflow_lite_support.python.task.vision import image_classifier
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.test import base_test
from tensorflow_lite_support.python.test import test_util

import unittest
import json


_MODEL_FLOAT = "mobilenet_v2_1.0_224.tflite"
_MODEL_QUANTIZED = "mobilenet_v1_0.25_224_quant.tflite"
_MODEL_AUTOML = "automl_labeler_model.tflite"


class ImageClassifierTest(parameterized.TestCase, base_test.BaseTestCase):

  def setUp(self):
    super().setUp()

  @staticmethod
  def create_classifier_from_options(model_file, **classification_options):
    base_options = task_options.BaseOptions(model_file=model_file)
    classifier_options = processor_options.ClassificationOptions(
      **classification_options)
    options = image_classifier.ImageClassifierOptions(
      base_options=base_options, classifier_options=classifier_options)
    classifier = image_classifier.ImageClassifier(options)

    return classifier

  @staticmethod
  def build_test_data(expected_categories):
    classifications = classifications_pb2.Classifications(head_index=0)
    classifications.classes.extend(
      [class_pb2.Category(**args) for args in expected_categories]
    )
    expected_result = classifications_pb2.ClassificationResult()
    expected_result.classifications.append(classifications)
    expected_result_dict = json.loads(json_format.MessageToJson(expected_result))

    return expected_result_dict

  @parameterized.parameters(
    (_MODEL_FLOAT,),
    (_MODEL_QUANTIZED,),
    (_MODEL_AUTOML,)
  )
  def test_create_from_options(self, model_name):
    # Get the model path from the test data directory
    model_file = test_util.get_test_data_path(model_name)

    # Creates with options containing model file successfully.
    base_options = task_options.BaseOptions(model_file=model_file)
    options = image_classifier.ImageClassifierOptions(base_options=base_options)
    image_classifier.ImageClassifier.create_from_options(options)

    # Creates the classifier with the `num_threads` option successfully
    base_options = task_options.BaseOptions(model_file=model_file, num_threads=4)
    options = image_classifier.ImageClassifierOptions(base_options=base_options)
    image_classifier.ImageClassifier.create_from_options(options)

    # Missing the model file.
    with self.assertRaisesRegex(
            TypeError,
            r"__init__\(\) missing 1 required positional argument: 'model_file'"):
      base_options = task_options.BaseOptions()

    # Invalid empty model path.
    with self.assertRaisesRegex(
            Exception,
            r"INVALID_ARGUMENT: ExternalFile must specify at least one of "
            r"'file_content', file_name' or 'file_descriptor_meta'\. "
            r"\[tflite::support::TfLiteSupportStatus='2'\]"):
      base_options = task_options.BaseOptions(model_file="")
      options = image_classifier.ImageClassifierOptions(base_options=base_options)
      image_classifier.ImageClassifier.create_from_options(options)

    # Invalid max results.
    with self.assertRaisesRegex(
            Exception,
            r"INVALID_ARGUMENT: Invalid `max_results` option: value must be != 0 "
            r"\[tflite::support::TfLiteSupportStatus='2'\]"):
      base_options = task_options.BaseOptions(model_file=model_file)
      classifier_options = processor_options.ClassificationOptions(
        max_results=0)
      options = image_classifier.ImageClassifierOptions(
        base_options=base_options,
        classifier_options=classifier_options)
      image_classifier.ImageClassifier.create_from_options(options)

  @parameterized.parameters(
    (_MODEL_FLOAT, ['foo'], ['bar']),
    (_MODEL_QUANTIZED, ['foo'], ['bar']),
    (_MODEL_AUTOML, ['foo'], ['bar'])
  )
  def test_combined_allowlist_and_denylist(self, model_name,
                                            label_allowlist,
                                            label_denylist):
    # Get the model path from the test data directory
    model_file = test_util.get_test_data_path(model_name)

    # Fails with combined allowlist and denylist
    with self.assertRaisesRegex(
            Exception,
            r"INVALID_ARGUMENT: `class_name_whitelist` and `class_name_blacklist` "
            r"are mutually exclusive options. "
            r"\[tflite::support::TfLiteSupportStatus='2'\]"):
      base_options = task_options.BaseOptions(model_file=model_file)
      classifier_options = processor_options.ClassificationOptions(
        label_allowlist=label_allowlist,
        label_denylist=label_denylist)
      options = image_classifier.ImageClassifierOptions(
        base_options=base_options,
        classifier_options=classifier_options)
      image_classifier.ImageClassifier.create_from_options(options)

  @parameterized.parameters(
    (
      _MODEL_FLOAT, 3,
      [
        {'index': 934, 'score': 0.7399742007255554, 'class_name': "cheeseburger"},
        {'index': 925, 'score': 0.026928534731268883, 'class_name': "guacamole"},
        {'index': 932, 'score': 0.025737214833498, 'class_name': "bagel"}
      ]
    ),
    (
      _MODEL_QUANTIZED, 3,
      [
        {'index': 934, 'score': 0.96484375, 'class_name': "cheeseburger"},
        {'index': 948, 'score': 0.0078125, 'class_name': "mushroom"},
        {'index': 924, 'score': 0.00390625, 'class_name': "plate"}
      ]
    ),
    (
      _MODEL_AUTOML, 3,
      [
        {'index': 2, 'score': 0.96484375, 'class_name': "roses"},
        {'index': 4, 'score': 0.01171875, 'class_name': "tulips"},
        {'index': 0, 'score': 0.0078125, 'class_name': "daisy"}
      ]
    )
  )
  def test_classify_model(self, model_name, max_results, expected_categories):
    # Get the model path from the test data directory.
    model_file = test_util.get_test_data_path(model_name)

    # Creates classifier.
    classifier = self.create_classifier_from_options(
      model_file,
      max_results=max_results
    )

    # Loads image.
    image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger.jpg"))

    # Classifies the input.
    image_result = classifier.classify(image, bounding_box=None)
    image_result_dict = json.loads(json_format.MessageToJson(image_result))

    # Builds test data.
    expected_result_dict = self.build_test_data(expected_categories)

    # Comparing results (classification w/o bounding box).
    self.assertDictAlmostEqual(image_result_dict, expected_result_dict, places=5)

  @parameterized.parameters(
    (
      _MODEL_FLOAT, 3,
      [
        {'index': 934, 'score': 0.8815076351165771, 'class_name': "cheeseburger"},
        {'index': 925, 'score': 0.019456762820482254, 'class_name': "guacamole"},
        {'index': 932, 'score': 0.012489477172493935, 'class_name': "bagel"}
      ]
    ),
    (
      _MODEL_QUANTIZED, 3,
      [
        {'index': 934, 'score': 0.96484375, 'class_name': "cheeseburger"},
        {'index': 935, 'score': 0.0078125, 'class_name': "hotdog"},
        {'index': 119, 'score': 0.0078125, 'class_name': "Dungeness crab"}
      ]
    ),
    (
      _MODEL_AUTOML, 3,
      [
        {'index': 2, 'score': 0.953125, 'class_name': "roses"},
        {'index': 0, 'score': 0.01171875, 'class_name': "daisy"},
        {'index': 1, 'score': 0.01171875, 'class_name': "dandelion"}
      ]
    )
  )
  def test_classify_model_with_bounding_box(self, model_name, max_results,
                                            expected_categories):
    # Get the model path from the test data directory.
    model_file = test_util.get_test_data_path(model_name)

    # Creates classifier.
    classifier = self.create_classifier_from_options(
      model_file,
      max_results=max_results
    )

    # Loads image.
    image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger.jpg"))

    # Bounding box in "burger.jpg" corresponding to "burger_crop.jpg".
    bounding_box = bounding_box_pb2.BoundingBox(
      origin_x=0, origin_y=0, width=400, height=325)

    # Classifies the input.
    image_result = classifier.classify(image, bounding_box)
    image_result_dict = json.loads(json_format.MessageToJson(image_result))

    # Builds test data.
    expected_result_dict = self.build_test_data(expected_categories)

    # Comparing results (classification w/ bounding box).
    self.assertDictAlmostEqual(image_result_dict, expected_result_dict, places=5)

  @parameterized.parameters(
    (
      _MODEL_FLOAT, 0.5,
      [{'index': 934, 'score': 0.7399742007255554, 'class_name': "cheeseburger"}]
    ),
    (
      _MODEL_QUANTIZED, 0.5,
      [{'index': 934, 'score': 0.96484375, 'class_name': "cheeseburger"}]
    )
  )
  def test_score_threshold_option(self, model_name, score_threshold, expected_categories):
    # Get the model path from the test data directory.
    model_file = test_util.get_test_data_path(model_name)

    # Creates classifier.
    classifier = self.create_classifier_from_options(
      model_file,
      score_threshold=score_threshold
    )

    # Loads image.
    image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger.jpg"))

    # Classifies the input.
    image_result = classifier.classify(image, bounding_box=None)
    image_result_dict = json.loads(json_format.MessageToJson(image_result))

    # Builds test data.
    expected_result_dict = self.build_test_data(expected_categories)

    # Comparing results (classification w/o bounding box).
    self.assertDictAlmostEqual(image_result_dict, expected_result_dict, places=5)

  @parameterized.parameters(
    (
      _MODEL_FLOAT, ['cheeseburger', 'guacamole'],
      [
        {'index': 934, 'score': 0.7399742007255554, 'class_name': "cheeseburger"},
        {'index': 925, 'score': 0.026928534731268883, 'class_name': "guacamole"}
      ]
    ),
    (
      _MODEL_QUANTIZED, ['cheeseburger', 'hotdog'],
      [
        {'index': 934, 'score': 0.96484375, 'class_name': "cheeseburger"},
        {'index': 935, 'score': 0.00390625, 'class_name': "hotdog"}
      ]
    )
  )
  def test_allowlist_option(self, model_name, label_allowlist, expected_categories):
    # Get the model path from the test data directory.
    model_file = test_util.get_test_data_path(model_name)

    # Creates classifier.
    classifier = self.create_classifier_from_options(
      model_file,
      label_allowlist=label_allowlist
    )

    # Loads image.
    image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger.jpg"))

    # Classifies the input.
    image_result = classifier.classify(image, bounding_box=None)
    image_result_dict = json.loads(json_format.MessageToJson(image_result))

    # Builds test data.
    expected_result_dict = self.build_test_data(expected_categories)

    # Comparing results (classification w/o bounding box).
    self.assertDictAlmostEqual(image_result_dict, expected_result_dict, places=5)

  def test_denylist_option(self, model_name=_MODEL_FLOAT):
    # Get the model path from the test data directory.
    model_file = test_util.get_test_data_path(model_name)

    # Creates classifier.
    classifier = self.create_classifier_from_options(
      model_file=model_file,
      score_threshold=0.01,
      label_denylist=['cheeseburger']
    )

    # Loads image
    image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger.jpg"))

    # Classifies the input.
    image_result = classifier.classify(image, bounding_box=None)
    image_result_dict = json.loads(json_format.MessageToJson(image_result))

    # Expected results.
    expected_categories = [
      {'index': 925, 'score': 0.026928534731268883, 'class_name': "guacamole"},
      {'index': 932, 'score': 0.025737214833498, 'class_name': "bagel"},
      {'index': 963, 'score': 0.010005592368543148, 'class_name': "meat loaf"}
    ]

    # Builds test data.
    expected_result_dict = self.build_test_data(expected_categories)

    # Comparing results (classification w/o bounding box).
    self.assertDictAlmostEqual(image_result_dict, expected_result_dict, places=5)


if __name__ == "__main__":
  unittest.main()
