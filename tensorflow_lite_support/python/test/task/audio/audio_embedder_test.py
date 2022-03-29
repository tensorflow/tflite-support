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
"""Tests for audio_embedder."""

import enum

from absl.testing import parameterized
# TODO(b/220067158): Change to import tensorflow and leverage tf.test once
# fixed the dependency issue.
import unittest

from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.processor.proto import embedding_options_pb2
from tensorflow_lite_support.python.task.audio import audio_embedder
from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.test import base_test
from tensorflow_lite_support.python.test import test_util


_BaseOptions = base_options_pb2.BaseOptions
_AudioEmbedder = audio_embedder.AudioEmbedder
_AudioEmbedderOptions = audio_embedder.AudioEmbedderOptions

_YAMNET_EMBEDDING_MODEL_FILE = 'yamnet_audio_classifier_with_metadata.tflite'


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class AudioEmbedderTest(parameterized.TestCase, base_test.BaseTestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_YAMNET_EMBEDDING_MODEL_FILE)

  def test_embed_default(self):
    # Create embedder.
    base_options = _BaseOptions(file_name=self.model_path)
    options = _AudioEmbedderOptions(
      base_options)
    embedder = _AudioEmbedder.create_from_options(options)

    # Load the input audio files.
    tensor0 = tensor_audio.TensorAudio.create_from_wav_file(
      test_util.get_test_data_path("speech.wav"),
      embedder.required_input_buffer_size)

    # Extract embeddings.
    result0 = embedder.embed(tensor0)

    # Check embedding sizes.
    def _check_embedding_size(result):
      self.assertLen(result.embeddings, 1)
      feature_vector = result.embeddings[0].feature_vector

      if options.embedding_options.quantize:
        self.assertLen(feature_vector.value_string, 521)
      else:
        self.assertLen(feature_vector.value_float, 521)

    _check_embedding_size(result0)


if __name__ == '__main__':
  unittest.main()
