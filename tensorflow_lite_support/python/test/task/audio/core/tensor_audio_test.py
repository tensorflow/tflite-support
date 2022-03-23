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
import unittest

from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.task.audio.core import audio_record
from tensorflow_lite_support.python.task.audio.core.pybinds import _pywrap_audio_buffer
from tensorflow_lite_support.python.test import test_util

_CppAudioFormat = _pywrap_audio_buffer.AudioFormat
_CppAudioBuffer = _pywrap_audio_buffer.AudioBuffer

_CHANNELS = 1
_SAMPLE_RATE = 16000
_SAMPLE_COUNT = 15600


class TensorAudioTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.test_audio_path = test_util.get_test_data_path('speech.wav')
    self.test_tensor_audio = tensor_audio.TensorAudio(
      _CppAudioFormat(_CHANNELS, _SAMPLE_RATE), _SAMPLE_COUNT)

  def test_create_from_wav_file(self):
    # Loads TensorAudio object from WAV file.
    tensor = tensor_audio.TensorAudio.create_from_wav_file(
      self.test_audio_path, _SAMPLE_COUNT)
    tensor_audio_format = tensor.format

    self.assertEqual(tensor_audio_format.channels, _CHANNELS)
    self.assertEqual(tensor_audio_format.sample_rate, _SAMPLE_RATE)
    self.assertEqual(tensor.sample_count, _SAMPLE_COUNT)
    self.assertIsInstance(tensor.data, _CppAudioBuffer)

  def test_load_from_array(self):
    # Loads audio data from a NumPy array.
    array = np.random.rand(_SAMPLE_COUNT, _CHANNELS).astype(np.float32)
    self.test_tensor_audio.load_from_array(array)

    # Gets the C++ AudioBuffer object.
    cpp_audio_buffer = self.test_tensor_audio.data
    cpp_audio_format = cpp_audio_buffer.audio_format

    self.assertEqual(cpp_audio_format.channels, _CHANNELS)
    self.assertEqual(cpp_audio_format.sample_rate, _SAMPLE_RATE)
    self.assertEqual(cpp_audio_buffer.buffer_size, _SAMPLE_COUNT)
    self.assertIsInstance(cpp_audio_buffer, _CppAudioBuffer)
    assert_almost_equal(cpp_audio_buffer.float_buffer, array)

  def test_load_from_array_fails_with_input_size_matching_sample_rate(self):
    # Fails loading audio data from a NumPy array with an input size
    # matching sample rate.
    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of samples. "
        f"Expect {_SAMPLE_COUNT}."):
      array = np.random.rand(_SAMPLE_RATE, _CHANNELS).astype(np.float32)
      self.test_tensor_audio.load_from_array(array)

  def test_load_from_array_fails_with_input_size_less_than_sample_rate(self):
    # Fails loading audio data from a NumPy array with an input size
    # less than sample rate.
    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of samples. "
        f"Expect {_SAMPLE_COUNT}."):
      input_buffer_size = 10000
      self.assertLess(input_buffer_size, _SAMPLE_RATE)
      array = np.random.rand(input_buffer_size, _CHANNELS).astype(np.float32)
      self.test_tensor_audio.load_from_array(array)

  def test_load_from_array_fails_with_invalid_number_of_channels(self):
    # Fails loading audio data from a NumPy array with an invalid
    # number of input channels.
    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of channels. "
        f"Expect {_CHANNELS}."):
      array = np.random.rand(_SAMPLE_COUNT, 2).astype(np.float32)
      self.test_tensor_audio.load_from_array(array)

  def test_load_from_array_fails_with_too_many_input_samples(self):
    # Fails loading audio data from a NumPy array with a sample count
    # exceeding TensorAudio's internal buffer capacity.
    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of samples. "
        f"Expect {_SAMPLE_COUNT}."):
      array = np.random.rand(20000, _CHANNELS).astype(np.float32)
      self.test_tensor_audio.load_from_array(array)

  def test_load_from_audio_record_fails_with_invalid_buffer_size(self):
    # Fails loading audio data from an AudioRecord instance having
    # a buffer size less than that of TensorAudio.
    with self.assertRaisesRegex(
        ValueError,
        "The audio record's buffer size cannot be smaller than the tensor "
        "audio's sample count."):
      record = audio_record.AudioRecord(_CHANNELS, _SAMPLE_RATE, 10000)
      self.test_tensor_audio.load_from_audio_record(record)

  def test_load_from_audio_record_fails_with_invalid_number_of_channels(self):
    # Fails loading audio data from an AudioRecord instance having
    # an invalid number of channels.
    with self.assertRaisesRegex(
        ValueError,
        f"The audio record's channel count doesn't match. "
        f"Expects {_CHANNELS} channel\(s\)."):
      record = audio_record.AudioRecord(2, _SAMPLE_RATE, _SAMPLE_COUNT)
      self.test_tensor_audio.load_from_audio_record(record)

  def test_load_from_audio_record_fails_with_invalid_sample_rate(self):
    # Fails loading audio data from an AudioRecord instance having
    # an invalid sample rate.
    with self.assertRaisesRegex(
        ValueError,
        f"The audio record's sampling rate doesn't match. "
        f"Expects {_SAMPLE_RATE}Hz."):
      record = audio_record.AudioRecord(_CHANNELS, 20000, _SAMPLE_COUNT)
      self.test_tensor_audio.load_from_audio_record(record)


if __name__ == '__main__':
  unittest.main()
