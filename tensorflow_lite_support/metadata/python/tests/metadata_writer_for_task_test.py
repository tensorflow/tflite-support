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

_AUDIO_CLASSIFICATION_MODEL = '../testdata/audio_classifier/yamnet_wavin_quantized_mel_relu6.tflite'
_AUDIO_EMBEDDING_MODEL = '../testdata/audio_embedder/yamnet_embedding.tflite'


class MetadataWriterForTaskTest(tf.test.TestCase):

  def test_initialize_and_populate(self):
    with mt.Writer(
        test_utils.load_file(_AUDIO_CLASSIFICATION_MODEL),
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

  def test_audio_embedder(self):
    with mt.Writer(
        test_utils.load_file(_AUDIO_EMBEDDING_MODEL),
        model_name='audio_embedder',
        model_description='Generate embedding for the input audio clip'
    ) as writer:
      out_dir = self.create_tempdir()
      writer.add_audio_input(sample_rate=16000, channels=1)
      writer.add_embedding_output()
      _, metadata_json = writer.populate(
          os.path.join(out_dir, 'model.tflite'),
          os.path.join(out_dir, 'metadata.json'))
      self.assertEqual(
          metadata_json, """{
  "name": "audio_embedder",
  "description": "Generate embedding for the input audio clip",
  "subgraph_metadata": [
    {
      "input_tensor_metadata": [
        {
          "name": "audio",
          "description": "Input audio clip to be processed.",
          "content": {
            "content_properties_type": "AudioProperties",
            "content_properties": {
              "sample_rate": 16000,
              "channels": 1
            }
          },
          "stats": {
          }
        }
      ],
      "output_tensor_metadata": [
        {
          "name": "embedding",
          "description": "Embedding vector of the input.",
          "content": {
            "content_properties_type": "FeatureProperties",
            "content_properties": {
            }
          },
          "stats": {
          }
        }
      ]
    }
  ],
  "min_parser_version": "1.3.0"
}
""")


if __name__ == '__main__':
  tf.test.main()
