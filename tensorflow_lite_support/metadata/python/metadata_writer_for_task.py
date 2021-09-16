# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Object oriented generic metadata writer for modular task API."""

import tempfile
from typing import Optional

from tensorflow_lite_support.metadata.python import metadata as _metadata
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_info
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_writer
from tensorflow_lite_support.metadata.python.metadata_writers import writer_utils


class Writer:
  """Generic object-oriented Metadata writer.

  Note that this API is experimental and is subject to changes. Also it only
  supports limited input and output tensor types for now. More types are being
  added.

  Example usage:

  The model has two inputs, audio and image respectively. And generates two
  outputs: classification and embedding.

  with open(model_path, 'rb') as f:
    with Writer(f.read(), 'model_name', 'model description') as writer:
      writer
        .add_audio_input(sample_rate=16000, channels=1)
        .add_image_input()
        .add_classification_head(class_names=['apple', 'banana'])
        .add_embedding_head()
        .populate('model.tflite', 'model.json')
  """

  def __init__(self, model_buffer: bytearray, model_name: str,
               model_description: str):
    self._model_buffer = model_buffer
    self._general_md = metadata_info.GeneralMd(
        name=model_name, description=model_description)
    self._input_mds = []
    self._output_mds = []
    self._associate_files = []

  def __enter__(self):
    self._temp_folder = tempfile.TemporaryDirectory()
    return self

  def __exit__(self, unused_exc_type, unused_exc_val, unused_exc_tb):
    self._temp_folder.cleanup()
    # Delete the attribute so that it errors out outside the `with` statement.
    delattr(self, '_temp_folder')

  def populate(self,
               tflite_path: Optional[str] = None,
               json_path: Optional[str] = None):
    """Writes the generated flatbuffer file / json metadata to disk.

    Note that you'll only need the tflite file for deployment. The JSON file
    is useful to help you understand what's in the metadata.

    Args:
      tflite_path: path to the tflite file.
      json_path: path to the JSON file.

    Returns:
      A tuple of (tflite_content_in_bytes, metdata_json_content)
    """
    tflite_content = None
    metadata_json_content = None

    writer = metadata_writer.MetadataWriter.create_from_metadata_info(
        model_buffer=self._model_buffer,
        general_md=self._general_md,
        input_md=self._input_mds,
        output_md=self._output_mds,
        associated_files=self._associate_files)

    if tflite_path:
      tflite_content = writer.populate()
      writer_utils.save_file(tflite_content, tflite_path)

    if json_path:
      displayer = _metadata.MetadataDisplayer.with_model_file(tflite_path)
      metadata_json_content = displayer.get_metadata_json()
      with open(json_path, 'w') as f:
        f.write(metadata_json_content)

    return (tflite_content, metadata_json_content)

  _INPUT_AUDIO_NAME = 'audio'
  _INPUT_AUDIO_DESCRIPTION = 'Input audio clip to be processed.'

  def add_audio_input(self,
                      sample_rate: int,
                      channels: int,
                      name: str = _INPUT_AUDIO_NAME,
                      description: str = _INPUT_AUDIO_DESCRIPTION):
    """Marks the next input tensor as an audio input."""
    # To make Task Library working properly, sample_rate, channels need to be
    # positive.
    if sample_rate <= 0:
      raise ValueError(
          'sample_rate should be positive, but got {}.'.format(sample_rate))
    if channels <= 0:
      raise ValueError(
          'channels should be positive, but got {}.'.format(channels))

    input_md = metadata_info.InputAudioTensorMd(
        name=name,
        description=description,
        sample_rate=sample_rate,
        channels=channels)
    self._input_mds.append(input_md)
    return self

  _OUTPUT_EMBEDDING_NAME = 'embedding'
  _OUTPUT_EMBEDDING_DESCRIPTION = 'Embedding vector of the input.'

  def add_embedding_output(self,
                           name: str = _OUTPUT_EMBEDDING_NAME,
                           description: str = _OUTPUT_EMBEDDING_DESCRIPTION):
    """Marks the next output tensor as embedding."""
    output_md = metadata_info.TensorMd(name=name, description=description)
    self._output_mds.append(output_md)
    return self
