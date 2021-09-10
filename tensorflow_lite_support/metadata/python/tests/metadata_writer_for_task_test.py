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
      writer.populate(
          os.path.join(out_dir, 'model.tflite'),
          os.path.join(out_dir, 'metadata.json'))
      self.assertJsonEqual(
          test_utils.load_file(os.path.join(out_dir, 'metadata.json'), 'r'),
          """{
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

  def test_audio_classifier(self):
    with mt.Writer(
        test_utils.load_file(_AUDIO_CLASSIFICATION_MODEL),
        model_name='audio_classifier',
        model_description='Classify the input audio clip') as writer:
      out_dir = self.create_tempdir()
      writer.add_audio_input(sample_rate=16000, channels=1)
      writer.add_classification_head(['soudn1', 'sound2'])
      writer.populate(
          os.path.join(out_dir, 'model.tflite'),
          os.path.join(out_dir, 'metadata.tflite'))
      self.assertEqual(
          test_utils.load_file(os.path.join(out_dir, 'metadata.tflite'), 'r'),
          """{
  "name": "audio_classifier",
  "description": "Classify the input audio clip",
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
          "name": "score",
          "description": "Score of the labels respectively",
          "content": {
            "content_properties_type": "FeatureProperties",
            "content_properties": {
            }
          },
          "stats": {
            "max": [
              1.0
            ],
            "min": [
              0.0
            ]
          },
          "associated_files": [
            {
              "name": "labels.txt",
              "description": "Labels for categories that the model can recognize.",
              "type": "TENSOR_AXIS_LABELS",
              "locale": "en"
            }
          ]
        }
      ]
    }
  ],
  "min_parser_version": "1.3.0"
}
""")

  def test_audio_classifier_with_locale(self):
    with mt.Writer(
        test_utils.load_file(_AUDIO_CLASSIFICATION_MODEL),
        model_name='audio_classifier',
        model_description='Classify the input audio clip') as writer:
      out_dir = self.create_tempdir()
      writer.add_audio_input(sample_rate=16000, channels=1)
      writer.add_classification_head(['/id1', '/id2'], {
          'en': ['sound1', 'sound2'],
          'fr': ['son1', 'son2']
      })
      writer.populate(
          os.path.join(out_dir, 'model.tflite'),
          os.path.join(out_dir, 'metadata.tflite'))
      self.assertEqual(
          test_utils.load_file(os.path.join(out_dir, 'metadata.tflite'), 'r'),
          """{
  "name": "audio_classifier",
  "description": "Classify the input audio clip",
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
          "name": "score",
          "description": "Score of the labels respectively",
          "content": {
            "content_properties_type": "FeatureProperties",
            "content_properties": {
            }
          },
          "stats": {
            "max": [
              1.0
            ],
            "min": [
              0.0
            ]
          },
          "associated_files": [
            {
              "name": "labels.txt",
              "description": "Labels for categories that the model can recognize.",
              "type": "TENSOR_AXIS_LABELS"
            },
            {
              "name": "labels_en.txt",
              "description": "Labels for categories that the model can recognize.",
              "type": "TENSOR_AXIS_LABELS",
              "locale": "en"
            },
            {
              "name": "labels_fr.txt",
              "description": "Labels for categories that the model can recognize.",
              "type": "TENSOR_AXIS_LABELS",
              "locale": "fr"
            }
          ]
        }
      ]
    }
  ],
  "min_parser_version": "1.3.0"
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
      writer.add_embedding_head()
      writer.populate(
          os.path.join(out_dir, 'model.tflite'),
          os.path.join(out_dir, 'metadata.json'))
      self.assertEqual(
          test_utils.load_file(os.path.join(out_dir, 'metadata.json'), 'r'),
          """{
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
