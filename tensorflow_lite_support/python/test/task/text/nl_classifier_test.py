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

import unittest
from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor import processor_options
from tensorflow_lite_support.python.task.text import nl_classifier
from tensorflow_lite_support.python.test import test_util

_ExternalFile = task_options.ExternalFile
_BaseOptions = task_options.BaseOptions
_NLClassifier = nl_classifier.NLClassifier
_NLClassifierOptions = nl_classifier.NLClassifierOptions
_NLClassificationOptions = processor_options.NLClassificationOptions
_ClassificationResult = processor_options.ClassificationResult
_Classifications = processor_options.Classifications
_Category = processor_options.Category

_REGEX_TOKENIZER_MODEL = 'test_model_nl_classifier_with_regex_tokenizer.tflite'
_POSITIVE_INPUT = "This is the best movie I’ve seen in recent years. " \
                  "Strongly recommend it!"
_EXPECTED_RESULTS_OF_POSITIVE_INPUT = _ClassificationResult(
  classifications=[_Classifications(
    categories=[_Category(
      index=None,
      score=0.48657345771789551,
      display_name='',
      class_name='Negative'
    ), _Category(
      index=None,
      score=0.51342660188674927,
      display_name='',
      class_name='Positive'
    )],
    head_index=0,
    head_name=None)
  ])


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class NLClassifierTest(parameterized.TestCase, unittest.TestCase):

  @parameterized.parameters(
      # Regex tokenizer model.
      (_REGEX_TOKENIZER_MODEL, ModelFileType.FILE_NAME, _POSITIVE_INPUT,
       _EXPECTED_RESULTS_OF_POSITIVE_INPUT))
  def test_classify_model(self, model_name, model_file_type, text,
                          expected_result):
    # Creates classifier.
    model_path = test_util.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      model_file = _ExternalFile(file_name=model_path)
      base_options = _BaseOptions(model_file=model_file)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, "rb") as f:
        model_content = f.read()
      model_file = _ExternalFile(file_content=model_content)
      base_options = _BaseOptions(model_file=model_file)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _NLClassifierOptions(base_options=base_options)
    classifier = _NLClassifier.create_from_options(options)
    # Classifies text using the given model.
    text_classification_result = classifier.classify(text)

    for i in range(len(text_classification_result.classifications)):
      a = text_classification_result.classifications[i]
      b = expected_result.classifications[i]
      self.assertEqual(a.head_index, b.head_index)
      self.assertEqual(len(a.categories), len(b.categories))
      for j in range(len(a.categories)):
        self.assertEqual(a.categories[j].index, b.categories[j].index)
        self.assertEqual(a.categories[j].class_name, b.categories[j].class_name)
        self.assertEqual(a.categories[j].display_name,
                         b.categories[j].display_name)
        self.assertAlmostEqual(a.categories[j].score,
                               b.categories[j].score, delta=1e-6)


if __name__ == "__main__":
  unittest.main()
