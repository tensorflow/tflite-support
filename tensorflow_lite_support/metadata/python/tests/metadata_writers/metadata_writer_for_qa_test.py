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
"""Tests for MetadataWriter for QA."""

import json
import os
import tempfile

import tensorflow as tf

from tensorflow.python.platform import resource_loader  # pylint: disable=g-direct-tensorflow-import
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_writer_for_qa as _writer

# Relative path to the script.
_MODEL = "../testdata/universal_sentence_encoder_qa/model.tflite"
_JSON_FILE = "../testdata/universal_sentence_encoder_qa/metadata.json"


class MetadataWriterForQaTest(tf.test.TestCase):

  def test_create_model(self):
    model_file = resource_loader.get_path_to_datafile(_MODEL)
    json_file = resource_loader.get_path_to_datafile(_JSON_FILE)

    with tempfile.TemporaryDirectory() as tempdir:
      output_file = os.path.join(tempdir, "model_with_metadata.tflite")
      populator = _writer.MetadataPopulatorForQA(model_file)
      populator.populate(output_file)
      self.assertTrue(os.path.exists(output_file))
      self.assertGreater(os.path.getsize(output_file), 0)

      metadata_dict = json.loads(populator.get_metadata_json())
      with tf.io.gfile.GFile(json_file) as f:
        expected_dict = json.load(f)
      self.assertDictEqual(metadata_dict, expected_dict)


if __name__ == "__main__":
  tf.test.main()
