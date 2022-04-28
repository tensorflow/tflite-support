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
from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.processor.proto import nl_classification_options_pb2
from tensorflow_lite_support.python.task.text import nl_classifier
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_pb2.BaseOptions
_NLClassifier = nl_classifier.NLClassifier
_NLClassifierOptions = nl_classifier.NLClassifierOptions

_MODEL = "test_model_nl_classifier_with_regex_tokenizer.tflite"

_POSITIVE_INPUT = "This is the best movie I’ve seen in recent years. " \
                  "Strongly recommend it!"


class NLClassifierTest(parameterized.TestCase, tf.test.TestCase):
  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_MODEL)

  def test_classify(self):
    # Create classifier.
    classifier = _NLClassifier.create_from_file(self.model_path)
    text_classification_result = classifier.classify(_POSITIVE_INPUT)
    print(text_classification_result)


if __name__ == "__main__":
  tf.test.main()
