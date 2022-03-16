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
"""Tests for tensor_audio."""
import numpy as np
from numpy.testing import assert_almost_equal
from absl.testing import parameterized
import unittest

from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.task.audio.core.pybinds import _pywrap_audio_buffer
from tensorflow_lite_support.python.test import test_util

_CppAudioFormat = _pywrap_audio_buffer.AudioFormat
_CppAudioBuffer = _pywrap_audio_buffer.AudioBuffer


class TensorAudioTest(parameterized.TestCase, unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.test_audio_path = test_util.get_test_data_path('speech.wav')

  def test_from_file(self):
    # Test data
    test_channels = 1
    test_sample_rate = 16000
    test_audio_format = _CppAudioFormat(test_channels, test_sample_rate)
    test_sample_count = 15600

    # Loads TensorAudio object from WAV file.
    tensor = tensor_audio.TensorAudio.from_wav_file(
      self.test_audio_path, 15600)
    tensor_audio_format = tensor.get_format()

    self.assertEqual(tensor.get_sample_count(), test_sample_count)
    self.assertEqual(tensor_audio_format.channels, test_audio_format.channels)
    self.assertEqual(
      tensor_audio_format.sample_rate, test_audio_format.sample_rate)
    self.assertIsInstance(tensor.get_data(), _CppAudioBuffer)

  def test_load_from_array(self):
    # Test data
    test_channels = 1
    test_sample_rate = 16000
    test_audio_format = _CppAudioFormat(
      test_channels, test_sample_rate)
    test_sample_count = 15600
    tensor = tensor_audio.TensorAudio(
      audio_format=test_audio_format, sample_count=test_sample_count)

    # Loads TensorAudio object from a NumPy array.
    array = np.random.rand(test_sample_count, test_channels).astype(np.float32)
    tensor.load_from_array(array)

    tensor_audio_data = tensor.get_data()
    tensor_audio_format = tensor_audio_data.audio_format

    self.assertEqual(tensor_audio_format.channels, test_channels)
    self.assertEqual(tensor_audio_format.sample_rate, test_sample_rate)
    self.assertEqual(tensor_audio_data.buffer_size, test_sample_count)
    self.assertIsInstance(tensor_audio_data, _CppAudioBuffer)
    assert_almost_equal(tensor_audio_data.float_buffer, array)

  def test_load_from_array_succeeds_with_input_size_matching_sample_rate(self):
    # Test data
    test_channels = 1
    test_sample_rate = test_sample_count = 16000
    tensor = tensor_audio.TensorAudio(
      audio_format=_CppAudioFormat(test_channels, test_sample_rate),
      sample_count=test_sample_count)

    # Loads TensorAudio object from a NumPy array with input sample count same
    # as test sample rate.
    array = np.random.rand(test_sample_rate, test_channels).astype(np.float32)
    tensor.load_from_array(array)
    tensor_audio_data = tensor.get_data()
    assert_almost_equal(tensor_audio_data.float_buffer, array)

  def test_load_from_array_fails_with_input_size_less_than_sample_rate(self):
    # Load TensorAudio object from a NumPy array.
    test_channels = 1
    test_sample_rate = 16000
    test_sample_count = 15600
    tensor = tensor_audio.TensorAudio(
      audio_format=_CppAudioFormat(test_channels, test_sample_rate),
      sample_count=test_sample_count)

    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of samples. "
        f"Expect {test_sample_count}."):
      input_buffer_size = 10000
      array = np.random.rand(input_buffer_size, 1).astype(np.float32)
      tensor.load_from_array(array)

  def test_load_from_array_fails_with_invalid_number_of_channels(self):
    # Load TensorAudio object from a NumPy array.
    test_channels = 1
    test_sample_rate = 16000
    test_sample_count = 15600
    tensor = tensor_audio.TensorAudio(
      audio_format=_CppAudioFormat(test_channels, test_sample_rate),
      sample_count=test_sample_count)

    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of channels. "
        f"Expect {test_channels}."):
      array = np.random.rand(test_sample_count, 2).astype(np.float32)
      tensor.load_from_array(array)

  def test_load_from_array_fails_with_too_many_input_samples(self):
    # Fails loading TensorAudio object from a NumPy array with a sample count
    # exceeding TensorAudio's internal buffer capacity.
    test_sample_count = 15000
    tensor = tensor_audio.TensorAudio(
      audio_format=_CppAudioFormat(1, 16000), sample_count=test_sample_count)

    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of samples. "
        f"Expect {test_sample_count}."):
      array = np.random.rand(20000, 1).astype(np.float32)
      tensor.load_from_array(array)

  def test_load_from_array_fails_with_invalid_input_channels(self):
    # Fails loading TensorAudio object from a NumPy array with input audio
    # containing an invalid number of channels.
    tensor = tensor_audio.TensorAudio(
      audio_format=_CppAudioFormat(1, 16000), sample_count=15000)

    with self.assertRaisesRegex(
        ValueError,
        r'Input audio contains an invalid number of channels.'):
      array = np.random.rand(15000, 4).astype(np.float32)
      tensor.load_from_array(array)


if __name__ == '__main__':
  unittest.main()
