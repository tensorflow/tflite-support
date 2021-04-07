# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for AudioClassifier.MetadataWriter."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata.python.metadata_writers import audio_classifier
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_info
from tensorflow_lite_support.metadata.python.tests.metadata_writers import test_utils

_FIXED_INPUT_SIZE_MODEL = "../testdata/audio_classifier/daredevil_sound_recognizer_320ms.tflite"
_DYNAMIC_INPUT_SIZE_MODEL = "../testdata/audio_classifier/yamnet_tfhub.tflite"
_LABEL_FILE = "../testdata/audio_classifier/labelmap.txt"
_SCORE_CALIBRATION_FILE = "../testdata/audio_classifier/score_calibration.txt"
_DEFAULT_SCORE_CALIBRATION_VALUE = 0.2
_JSON_FOR_INFERENCE_DYNAMIC = "../testdata/audio_classifier/yamnet_tfhub.json"
_JSON_FOR_INFERENCE_FIXED = "../testdata/audio_classifier/daredevil_sound_recognizer_320ms.json"
_JSON_DEFAULT = "../testdata/audio_classifier/daredevil_sound_recognizer_320ms_default.json"
_SAMPLE_RATE = 2
_CHANNELS = 1


class MetadataWriterTest(tf.test.TestCase):

  def test_create_for_inference_should_succeed_dynaamic_input_shape_model(self):
    writer = audio_classifier.MetadataWriter.create_for_inference(
        test_utils.load_file(_DYNAMIC_INPUT_SIZE_MODEL), _SAMPLE_RATE,
        _CHANNELS, [_LABEL_FILE],
        metadata_info.ScoreCalibrationMd(
            _metadata_fb.ScoreTransformationType.LOG,
            _DEFAULT_SCORE_CALIBRATION_VALUE, _SCORE_CALIBRATION_FILE))

    metadata_json = writer.get_metadata_json()
    expected_json = test_utils.load_file(_JSON_FOR_INFERENCE_DYNAMIC, "r")
    self.assertEqual(metadata_json, expected_json)

  def test_create_for_inference_should_succeed_with_fixed_input_shape_model(
      self):
    writer = audio_classifier.MetadataWriter.create_for_inference(
        test_utils.load_file(_FIXED_INPUT_SIZE_MODEL), _SAMPLE_RATE, _CHANNELS,
        [_LABEL_FILE],
        metadata_info.ScoreCalibrationMd(
            _metadata_fb.ScoreTransformationType.LOG,
            _DEFAULT_SCORE_CALIBRATION_VALUE, _SCORE_CALIBRATION_FILE))

    metadata_json = writer.get_metadata_json()
    expected_json = test_utils.load_file(_JSON_FOR_INFERENCE_FIXED, "r")
    self.assertEqual(metadata_json, expected_json)

  def test_create_from_metadata_info_by_default_should_succeed(self):
    writer = audio_classifier.MetadataWriter.create_from_metadata_info(
        test_utils.load_file(_FIXED_INPUT_SIZE_MODEL))

    metadata_json = writer.get_metadata_json()
    expected_json = test_utils.load_file(_JSON_DEFAULT, "r")
    self.assertEqual(metadata_json, expected_json)


class MetadataWriterSampleRateTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "negative",
          "wrong_sample_rate": -1
      }, {
          "testcase_name": "zero",
          "wrong_sample_rate": 0
      })
  def test_create_for_inference_fails_with_wrong_sample_rate(
      self, wrong_sample_rate):

    with self.assertRaises(ValueError) as error:
      audio_classifier.MetadataWriter.create_for_inference(
          test_utils.load_file(_DYNAMIC_INPUT_SIZE_MODEL), wrong_sample_rate,
          _CHANNELS, [_LABEL_FILE],
          metadata_info.ScoreCalibrationMd(
              _metadata_fb.ScoreTransformationType.LOG,
              _DEFAULT_SCORE_CALIBRATION_VALUE, _SCORE_CALIBRATION_FILE))

    self.assertEqual(
        "sample_rate should be positive, but got {}.".format(wrong_sample_rate),
        str(error.exception))


class MetadataWriterChannelsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          "testcase_name": "negative",
          "wrong_channels": -1
      }, {
          "testcase_name": "zero",
          "wrong_channels": 0
      })
  def test_create_for_inference_fails_with_wrong_channels(self, wrong_channels):

    with self.assertRaises(ValueError) as error:
      audio_classifier.MetadataWriter.create_for_inference(
          test_utils.load_file(_DYNAMIC_INPUT_SIZE_MODEL), _SAMPLE_RATE,
          wrong_channels, [_LABEL_FILE],
          metadata_info.ScoreCalibrationMd(
              _metadata_fb.ScoreTransformationType.LOG,
              _DEFAULT_SCORE_CALIBRATION_VALUE, _SCORE_CALIBRATION_FILE))

    self.assertEqual(
        "channels should be positive, but got {}.".format(wrong_channels),
        str(error.exception))


if __name__ == "__main__":
  tf.test.main()
