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
"""Tests for tensorflow_lite_support.metadata.metadata_writer_for_task."""

import os
import tensorflow as tf
from tensorflow_lite_support.metadata.python import metadata_writer_for_task as mt
from tensorflow_lite_support.metadata.python.tests.metadata_writers import test_utils

_AUDIO_MODEL = '../testdata/audio_classifier/yamnet_wavin_quantized_mel_relu6.tflite'


class MetadataWriterForTaskTest(tf.test.TestCase):

  def test_initialize_and_populate(self):
    with mt.Writer(
        test_utils.load_file(_AUDIO_MODEL),
        model_name='my_audio_model',
        model_description='my_description') as writer:
      out_dir = self.create_tempdir()
      _, metadata_json = writer.populate(
          os.path.join(out_dir, 'model.tflite'),
          os.path.join(out_dir, 'metadata.json'))
      self.assertJsonEqual(
          metadata_json, """{
  "name": "my_audio_model",
  "description": "my_description",
  "subgraph_metadata": [
    {
      "input_tensor_metadata": [
        {
          "name": "waveform_binary"
        }
      ],
      "output_tensor_metadata": [
        {
          "name": "tower0/network/layer32/final_output"
        }
      ]
    }
  ],
  "min_parser_version": "1.0.0"
}
""")


if __name__ == '__main__':
  tf.test.main()
