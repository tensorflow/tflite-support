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
"""Tests for nl_classifier."""

import enum

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_lite_support.python.task.core import base_options as base_options_module
from tensorflow_lite_support.python.task.processor.proto import class_pb2
from tensorflow_lite_support.python.task.processor.proto import classifications_pb2
from tensorflow_lite_support.python.task.processor.proto import classification_options_pb2
from tensorflow_lite_support.python.task.text import nl_classifier
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_module.BaseOptions
_NLClassifier = nl_classifier.NLClassifier
_Category = class_pb2.Category
_Classifications = classifications_pb2.Classifications
_ClassificationResult = classifications_pb2.ClassificationResult
_NLClassifierOptions = nl_classifier.NLClassifierOptions
_ClassificationOptions = classification_options_pb2.ClassificationOptions


_REGEX_TOKENIZER_MODEL = 'test_model_nl_classifier_with_regex_tokenizer.tflite'
_POSITIVE_INPUT = "This is the best movie I’ve seen in recent years. " \
                  "Strongly recommend it!"
_EXPECTED_RESULTS_OF_POSITIVE_INPUT = _ClassificationResult(
  classifications=[
    _Classifications(
      categories=[
        _Category(
          index=0,
          score=0.513427,
          display_name='',
          category_name='Positive'
        ),
        _Category(
          index=0,
          score=0.486573,
          display_name='',
          category_name='Negative'),
      ],
      head_index=0,
      head_name=''
    )])

_INPUT_STR = "hello"
_ALLOW_LIST = ['Positive', 'Negative']
_DENY_LIST = ['Negative']
_SCORE_THRESHOLD = 0.5
_MAX_RESULTS = 1


def _create_classifier_from_options(base_options, **classification_options):
  classification_options = classification_options_pb2.ClassificationOptions(
    **classification_options)
  options = _NLClassifierOptions(
    base_options=base_options, classification_options=classification_options)
  classifier = _NLClassifier.create_from_options(options)
  return classifier


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class NLClassifierTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_REGEX_TOKENIZER_MODEL)

  @parameterized.parameters(
      # Regex tokenizer model.
      (_REGEX_TOKENIZER_MODEL, ModelFileType.FILE_NAME, _POSITIVE_INPUT,
       _EXPECTED_RESULTS_OF_POSITIVE_INPUT),
      (_REGEX_TOKENIZER_MODEL, ModelFileType.FILE_CONTENT, _POSITIVE_INPUT,
       _EXPECTED_RESULTS_OF_POSITIVE_INPUT))
  def test_classify_model(self, model_name, model_file_type, text,
                          expected_classification_result):
    # Creates classifier.
    model_path = test_util.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, "rb") as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _NLClassifierOptions(base_options=base_options)
    classifier = _NLClassifier.create_from_options(options)
    # Classifies text using the given model.
    text_classification_result = classifier.classify(text)
    self.assertProtoEquals(text_classification_result.to_pb2(),
                           expected_classification_result.to_pb2())

  def test_max_results_option(self):
    # Creates classifier.
    base_options = _BaseOptions(file_name=self.model_path)

    classifier = _create_classifier_from_options(
      base_options, max_results=_MAX_RESULTS)

    # Classifies the input.
    text_result = classifier.classify(_INPUT_STR)
    categories = text_result.classifications[0].categories

    self.assertLessEqual(
      len(categories), _MAX_RESULTS, 'Too many results returned.')

  def test_score_threshold_option(self):
    # Creates classifier.
    base_options = _BaseOptions(file_name=self.model_path)

    classifier = _create_classifier_from_options(
      base_options, score_threshold=_SCORE_THRESHOLD)

    # Classifies the input.
    text_result = classifier.classify(_INPUT_STR)
    categories = text_result.classifications[0].categories

    for category in categories:
      self.assertGreaterEqual(
        category.score, _SCORE_THRESHOLD,
        f'Classification with score lower than threshold found. {category}')

  def test_allowlist_option(self):
    # Creates classifier.
    base_options = _BaseOptions(file_name=self.model_path)

    classifier = _create_classifier_from_options(
      base_options, category_name_allowlist=_ALLOW_LIST)

    # Classifies the input.
    text_result = classifier.classify(_INPUT_STR)
    categories = text_result.classifications[0].categories

    for category in categories:
      label = category.category_name
      self.assertIn(label, _ALLOW_LIST,
                    f'Label {label} found but not in label allow list')

  def test_denylist_option(self):
    # Creates classifier.
    base_options = _BaseOptions(file_name=self.model_path)

    classifier = _create_classifier_from_options(
      base_options, score_threshold=0.01, category_name_denylist=_DENY_LIST)

    # Classifies the input.
    text_result = classifier.classify(_INPUT_STR)
    categories = text_result.classifications[0].categories

    for category in categories:
      label = category.category_name
      self.assertNotIn(label, _DENY_LIST,
                       f'Label {label} found but in deny list.')

  def test_combined_allowlist_and_denylist(self):
    # Fails with combined allowlist and denylist
    with self.assertRaisesRegex(
        ValueError,
        r'`category_name_allowlist` and `category_name_denylist` are mutually '
        r'exclusive options.'):
      base_options = _BaseOptions(file_name=self.model_path)
      classification_options = classification_options_pb2.ClassificationOptions(
        category_name_allowlist=['foo'], category_name_denylist=['bar'])
      options = _NLClassifierOptions(
        base_options=base_options,
        classification_options=classification_options)
      _NLClassifier.create_from_options(options)


if __name__ == "__main__":
  tf.test.main()
