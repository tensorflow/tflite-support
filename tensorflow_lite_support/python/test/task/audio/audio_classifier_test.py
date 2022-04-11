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
"""Tests for audio_classifier."""

import enum
import json

from absl.testing import parameterized

from google.protobuf import json_format
import unittest
from tensorflow_lite_support.python.task.audio import audio_classifier
from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.processor.proto import class_pb2
from tensorflow_lite_support.python.task.processor.proto import classification_options_pb2
from tensorflow_lite_support.python.task.processor.proto import classifications_pb2
from tensorflow_lite_support.python.test import base_test
from tensorflow_lite_support.python.test import test_util

# TODO(b/220067158): Change to import tensorflow and leverage tf.test once
# fixed the dependency issue.

_BaseOptions = base_options_pb2.BaseOptions
_AudioClassifier = audio_classifier.AudioClassifier
_AudioClassifierOptions = audio_classifier.AudioClassifierOptions

_FIXED_INPUT_SIZE_MODEL_FILE = 'yamnet_audio_classifier_with_metadata.tflite'
_SPEECH_AUDIO_FILE = 'speech.wav'
_FIXED_INPUT_SIZE_MODEL_CLASSIFICATIONS = {
    'scores': [{
        'index': 0,
        'score': 0.91796875,
        'class_name': 'Speech'
    }, {
        'index': 500,
        'score': 0.05859375,
        'class_name': 'Inside, small room'
    }, {
        'index': 494,
        'score': 0.01367188,
        'class_name': 'Silence'
    }]
}
_ACCEPTABLE_ERROR_RANGE = 0.005


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


def _create_classifier_from_options(base_options, **classification_options):
  classification_options = classification_options_pb2.ClassificationOptions(
      **classification_options)
  options = _AudioClassifierOptions(
      base_options=base_options, classification_options=classification_options)
  classifier = _AudioClassifier.create_from_options(options)
  return classifier


def _build_test_data(classifications):
  expected_result = classifications_pb2.ClassificationResult()

  for index, (head_name, categories) in enumerate(classifications.items()):
    classifications = classifications_pb2.Classifications(
        head_index=index, head_name=head_name)
    classifications.classes.extend(
        [class_pb2.Category(**args) for args in categories])
    expected_result.classifications.append(classifications)

  expected_result_dict = json.loads(json_format.MessageToJson(expected_result))

  return expected_result_dict


class AudioClassifierTest(parameterized.TestCase, base_test.BaseTestCase):

  @parameterized.parameters(
      (_FIXED_INPUT_SIZE_MODEL_FILE, ModelFileType.FILE_NAME,
       _SPEECH_AUDIO_FILE, 3, _FIXED_INPUT_SIZE_MODEL_CLASSIFICATIONS),
      (_FIXED_INPUT_SIZE_MODEL_FILE, ModelFileType.FILE_CONTENT,
       _SPEECH_AUDIO_FILE, 3, _FIXED_INPUT_SIZE_MODEL_CLASSIFICATIONS),
  )
  def test_classify_model(self, model_name, model_file_type, audio_file_name,
                          max_results, expected_classifications):
    # Creates classifier.
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

    classifier = _create_classifier_from_options(
        base_options, max_results=max_results)

    # Load the input audio file.
    test_audio_path = test_util.get_test_data_path(audio_file_name)
    tensor = tensor_audio.TensorAudio.create_from_wav_file(
        test_audio_path, classifier.required_input_buffer_size)

    # Classifies the input.
    audio_result = classifier.classify(tensor)
    audio_result_dict = json.loads(json_format.MessageToJson(audio_result))

    # Builds test data.
    expected_result_dict = _build_test_data(expected_classifications)

    # Comparing results.
    self.assertDeepAlmostEqual(
        audio_result_dict, expected_result_dict, delta=_ACCEPTABLE_ERROR_RANGE)


if __name__ == '__main__':
  unittest.main()
