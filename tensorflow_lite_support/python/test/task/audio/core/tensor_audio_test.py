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
import unittest
from unittest import mock

import numpy as np
from numpy import testing

from tensorflow_lite_support.python.task.audio.core import tensor_audio
from tensorflow_lite_support.python.task.audio.core import audio_record
from tensorflow_lite_support.python.task.audio.core.pybinds import _pywrap_audio_buffer
from tensorflow_lite_support.python.test import test_util

_CppAudioFormat = _pywrap_audio_buffer.AudioFormat

_CHANNELS = 1
_SAMPLE_RATE = 16000
_BUFFER_SIZE = 15600


class TensorAudioTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.test_audio_path = test_util.get_test_data_path('speech.wav')
    self.test_tensor_audio = tensor_audio.TensorAudio(
      _CppAudioFormat(_CHANNELS, _SAMPLE_RATE), _BUFFER_SIZE)

  def test_create_from_wav_file_succeeds(self):
    # Loads TensorAudio object from WAV file.
    tensor = tensor_audio.TensorAudio.create_from_wav_file(
      self.test_audio_path, _BUFFER_SIZE)
    tensor_audio_format = tensor.format

    self.assertEqual(tensor_audio_format.channels, _CHANNELS)
    self.assertEqual(tensor_audio_format.sample_rate, _SAMPLE_RATE)
    self.assertEqual(tensor.buffer_size, _BUFFER_SIZE)
    self.assertIsInstance(tensor.buffer, np.ndarray)
    self.assertEqual(tensor.buffer[-1], -0.09640503)

  def test_create_from_wav_file_fails_with_empty_file_path(self):
    # Fails loading TensorAudio object from WAV file.
    with self.assertRaisesRegex(
        Exception,
        "INVALID_ARGUMENT: Data too short when trying to read string"):
      tensor_audio.TensorAudio.create_from_wav_file('', _BUFFER_SIZE)

  def test_load_from_array_succeeds(self):
    # Loads audio data from a NumPy array.
    array = np.random.rand(_BUFFER_SIZE, _CHANNELS).astype(np.float32)
    self.test_tensor_audio.load_from_array(array)

    audio_buffer = self.test_tensor_audio.buffer
    audio_format = self.test_tensor_audio.format

    self.assertEqual(audio_format.channels, _CHANNELS)
    self.assertEqual(audio_format.sample_rate, _SAMPLE_RATE)
    self.assertEqual(self.test_tensor_audio.buffer_size, _BUFFER_SIZE)
    self.assertIsInstance(audio_buffer, np.ndarray)
    testing.assert_almost_equal(audio_buffer, array)

  def test_load_from_array_fails_with_input_size_matching_sample_rate(self):
    # Fails loading audio data from a NumPy array with an input size
    # matching sample rate.
    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of samples. "
        f"Expect {_BUFFER_SIZE}."):
      array = np.random.rand(_SAMPLE_RATE, _CHANNELS).astype(np.float32)
      self.test_tensor_audio.load_from_array(array)

  def test_load_from_array_fails_with_input_size_less_than_sample_rate(self):
    # Fails loading audio data from a NumPy array with an input size
    # less than sample rate.
    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of samples. "
        f"Expect {_BUFFER_SIZE}."):
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
      array = np.random.rand(_BUFFER_SIZE, 2).astype(np.float32)
      self.test_tensor_audio.load_from_array(array)

  def test_load_from_array_fails_with_too_many_input_samples(self):
    # Fails loading audio data from a NumPy array with a sample count
    # exceeding TensorAudio's internal buffer capacity.
    with self.assertRaisesRegex(
        ValueError,
        f"Input audio contains an invalid number of samples. "
        f"Expect {_BUFFER_SIZE}."):
      array = np.random.rand(20000, _CHANNELS).astype(np.float32)
      self.test_tensor_audio.load_from_array(array)

  @mock.patch("sounddevice.InputStream", return_value=mock.MagicMock())
  def test_load_from_audio_record(self, mock_input_stream):
    record = audio_record.AudioRecord(_CHANNELS, _SAMPLE_RATE, _BUFFER_SIZE)

    # Get AudioRecord's audio callback function.
    _, mock_input_stream_init_args = mock_input_stream.call_args
    callback_fn = mock_input_stream_init_args["callback"]

    # Create dummy data to feed to the AudioRecord instance.
    chunk_size = int(_BUFFER_SIZE * 0.5)
    input_data = []
    for _ in range(3):
      dummy_data = np.random.rand(chunk_size, _CHANNELS).astype(float)
      input_data.append(dummy_data)
      callback_fn(dummy_data)
    expected_data = np.concatenate(input_data[-2:])

    # Load audio data into TensorAudio from the AudioRecord instance.
    self.test_tensor_audio.load_from_audio_record(record)

    # Assert read all data in the float buffer.
    testing.assert_almost_equal(self.test_tensor_audio.buffer, expected_data)

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
      record = audio_record.AudioRecord(2, _SAMPLE_RATE, _BUFFER_SIZE)
      self.test_tensor_audio.load_from_audio_record(record)

  def test_load_from_audio_record_fails_with_invalid_sample_rate(self):
    # Fails loading audio data from an AudioRecord instance having
    # an invalid sample rate.
    with self.assertRaisesRegex(
        ValueError,
        f"The audio record's sampling rate doesn't match. "
        f"Expects {_SAMPLE_RATE}Hz."):
      record = audio_record.AudioRecord(_CHANNELS, 20000, _BUFFER_SIZE)
      self.test_tensor_audio.load_from_audio_record(record)


if __name__ == '__main__':
  unittest.main()
