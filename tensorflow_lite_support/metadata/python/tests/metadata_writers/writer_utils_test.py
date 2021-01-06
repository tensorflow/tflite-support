# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for wrtier util methods."""

import tensorflow as tf

from tensorflow_lite_support.metadata import schema_py_generated as _schema_fb
from tensorflow_lite_support.metadata.python.metadata_writers import writer_utils
from tensorflow_lite_support.metadata.python.tests.metadata_writers import test_utils

_FLOAT_TYPE = _schema_fb.TensorType.FLOAT32
_UINT8_TYPE = _schema_fb.TensorType.UINT8
# mobilebert_float.tflite has 1 input tensor and 4 output tensors.
_MODEL_NAME = "../testdata/object_detector/ssd_mobilenet_v1.tflite"
_EXPECTED_INPUT_TYPES = _UINT8_TYPE
_EXPECTED_OUTPUT_TYPES = (_FLOAT_TYPE, _FLOAT_TYPE, _FLOAT_TYPE, _FLOAT_TYPE)


class WriterUtilsTest(tf.test.TestCase):

  def test_get_input_tensor_types(self):
    tensor_types = writer_utils.get_input_tensor_types(
        model_buffer=test_utils.load_file(_MODEL_NAME))
    self.assertEqual(tensor_types, [_EXPECTED_INPUT_TYPES])

  def test_get_output_tensor_types(self):
    tensor_types = writer_utils.get_output_tensor_types(
        model_buffer=test_utils.load_file(_MODEL_NAME))
    self.assertEqual(tensor_types, list(_EXPECTED_OUTPUT_TYPES))

  def test_save_and_load_file(self):
    expected_file_bytes = b"This is a test file."
    file_path = self.create_tempfile().full_path

    writer_utils.save_file(expected_file_bytes, file_path)
    file_bytes = writer_utils.load_file(file_path)
    self.assertEqual(file_bytes, expected_file_bytes)


if __name__ == "__main__":
  tf.test.main()
