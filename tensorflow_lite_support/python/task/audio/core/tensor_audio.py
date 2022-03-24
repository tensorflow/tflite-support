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
"""TensorAudio class."""

import numpy as np

from tensorflow_lite_support.python.task.audio.core.pybinds import _pywrap_audio_buffer
from tensorflow_lite_support.python.task.audio.core import audio_record

_CppAudioFormat = _pywrap_audio_buffer.AudioFormat
_LoadAudioBufferFromFile = _pywrap_audio_buffer.LoadAudioBufferFromFile


class TensorAudio(object):
  """A wrapper class to store the input audio."""

  def __init__(self,
               audio_format: _CppAudioFormat,
               buffer_size: int) -> None:
    """Initializes the `TensorAudio` object.

    Args:
      audio_format: format of the audio.
      buffer_size: buffer size of the audio.
    """
    self._format = audio_format
    self._buffer_size = buffer_size
    self._buffer = np.zeros(
        [self._buffer_size, self._format.channels], dtype=np.float32)

  def clear(self):
    """Clear the internal buffer and fill it with zeros."""
    self._buffer.fill(0)

  @classmethod
  def create_from_wav_file(cls,
                           file_name: str,
                           buffer_size: int) -> "TensorAudio":
    """Creates `TensorAudio` object from the WAV file.

    Args:
      file_name: WAV file name.
      buffer_size: Required input buffer size. The number of samples that the
      C++ AudioBuffer object can store. If the WAV file contains more samples
      than `buffer_size`, only the samples at the beginning of the WAV file will
      be loaded.

    Returns:
      `TensorAudio` object.

    Raises:
      status.StatusNotOk if the audio file can't be decoded.
    """
    # TODO(b/220931229): Raise RuntimeError instead of status.StatusNotOk.
    # Need to import the module to catch this error:
    # `from pybind11_abseil import status`
    # see https://github.com/pybind/pybind11_abseil#abslstatusor.
    audio = _LoadAudioBufferFromFile(
      file_name, buffer_size, np.zeros([buffer_size]))
    tensor = TensorAudio(audio.audio_format, audio.buffer_size)
    tensor.load_from_array(np.array(audio.float_buffer, copy=False))
    return tensor

  def load_from_audio_record(self, record: audio_record.AudioRecord) -> None:
    """Loads audio data from an AudioRecord instance.
    Args:
      record: An AudioRecord instance.
    Raises:
      ValueError: Raised if the audio record's config is invalid.
    """
    if record.buffer_size < self._buffer_size:
      raise ValueError(
        "The audio record's buffer size cannot be smaller than the tensor "
        "audio's sample count.")

    if record.channels != self._format.channels:
      raise ValueError(
        f"The audio record's channel count doesn't match. "
        f"Expects {self._format.channels} channel(s).")

    if record.sampling_rate != self._format.sample_rate:
      raise ValueError(
        f"The audio record's sampling rate doesn't match. "
        f"Expects {self._format.sample_rate}Hz.")

    # Load audio data from the AudioRecord instance.
    data = record.read(self._buffer_size)
    self.load_from_array(data.astype(np.float32))

  def load_from_array(self, src: np.ndarray) -> None:
    """Loads audio data from a NumPy array.

    Args:
      src: A NumPy array contains the input audio.

    Raises:
      ValueError if the input audio is too large or if it contains an invalid
      number of channels.
    """
    if len(src) != len(self._buffer):
      raise ValueError(
        f"Input audio contains an invalid number of samples. "
        f"Expect {len(self._buffer)}.")
    elif src.shape[1] != self._format.channels:
      raise ValueError(
        f"Input audio contains an invalid number of channels. "
        f"Expect {self._format.channels}.")

    self._buffer = src

  @property
  def format(self) -> _CppAudioFormat:
    """Gets the audio format of the audio."""
    return self._format

  @property
  def buffer_size(self) -> int:
    """Gets the sample count of the audio."""
    return self._buffer_size

  @property
  def buffer(self) -> np.ndarray:
    """Gets the internal buffer."""
    return self._buffer
