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

import collections
import os
import tempfile
from typing import Optional, List, Dict

from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata.python import metadata as _metadata
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_info
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_writer
from tensorflow_lite_support.metadata.python.metadata_writers import writer_utils

CalibrationParameter = collections.namedtuple(
    'CalibrationParameter', ['scale', 'slope', 'offset', 'min_score'])


class Writer:
  """Generic object-oriented Metadata writer.

  Note that this API is experimental and is subject to changes. Also it only
  supports limited input and output tensor types for now. More types are being
  added.

  Example usage:

  The model has two inputs, audio and image respectively. And generates two
  outputs: classification and embedding.

  with open(model_path, 'rb') as f:
    with MobileIca(f.read(), 'model_name', 'model description') as writer:
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
      The current Writer instance to allow chained operation.
    """
    writer = metadata_writer.MetadataWriter.create_from_metadata_info(
        model_buffer=self._model_buffer,
        general_md=self._general_md,
        input_md=self._input_mds,
        output_md=self._output_mds,
        associated_files=self._associate_files)

    if tflite_path:
      writer_utils.save_file(writer.populate(), tflite_path)

    if json_path:
      displayer = _metadata.MetadataDisplayer.with_model_file(tflite_path)
      with open(json_path, 'w') as f:
        f.write(displayer.get_metadata_json())

    return self

  def _export_labels(self, filename: str, index_to_label: List[str]):
    filepath = os.path.join(self._temp_folder.name, filename)
    with open(filepath, 'w') as f:
      f.write('\n'.join(index_to_label))
    self._associate_files.append(filepath)
    return filepath

  def _output_tensor_type(self, idx):
    return writer_utils.get_output_tensor_types(self._model_buffer)[idx]

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

  def add_embedding_head(self,
                         name: str = _OUTPUT_EMBEDDING_NAME,
                         description: str = _OUTPUT_EMBEDDING_DESCRIPTION):
    """Marks the next output tensor as embedding."""
    output_md = metadata_info.TensorMd(name=name, description=description)
    self._output_mds.append(output_md)
    return self

  def _export_calibration_file(self, filename: str,
                               calibrations: List[CalibrationParameter]):
    """Store calibration data in a csv file."""
    filepath = os.path.join(self._temp_folder.name, filename)
    with open(filepath, 'w') as f:
      for idx, item in enumerate(calibrations):
        if idx != 0:
          f.write('\n')
        if item:
          scale, slope, offset, min_score = item
          if all(x is not None for x in item):
            f.write(f'{scale},{slope},{offset},{min_score}')
          elif all(x is not None for x in item[:3]):
            f.write(f'{scale},{slope},{offset}')
          else:
            raise ValueError('scale, slope and offset values can not be set to '
                             'None.')
          self._associate_files.append(filepath)
    return filepath

  _OUTPUT_CLASSIFICATION_NAME = 'score'
  _OUTPUT_CLASSIFICATION_DESCRIPTION = 'Score of the labels respectively'

  def add_classification_head(
      self,
      class_names: List[str],
      locale_to_labels: Optional[Dict[str, List[str]]] = None,
      score_calibration: Optional[List[CalibrationParameter]] = None,
      score_transformation_type: Optional[
          _metadata_fb.ScoreTransformationType] = None,
      default_calibrated_score: Optional[float] = None,
      name=_OUTPUT_CLASSIFICATION_NAME,
      description=_OUTPUT_CLASSIFICATION_DESCRIPTION):
    """Marks model's next output tensor as a classification head.

    Example usage:
    writer.add_classification_head(
      ['/m/011l78"', '/m/031d23'],
      {'en': ['cat', 'dog'],
       'fr': ['chat', 'chien']})

    If the class names are human readable, then `locale_to_labels` can be
    skipped. And the class names will be used as display name with locale='en'

    For example:
    writer.add_classification_head(['cat', 'dog'])

    Args:
      class_names: a list of machine readable labels. The labels will be used as
        `class_name` during deployment.
      locale_to_labels: a mapping from locale to label list. `en` is the default
        locale used by the Task Library.
      score_calibration: a list of calibration parameters.
      score_transformation_type: type of the function used for transforming the
        uncalibrated score before applying score calibration.
      default_calibrated_score: the default calibrated score to apply if the
        uncalibrated score is below min_score or if no parameters were specified
        for a given index.
      name: Metadata name of the tensor. Note that this is different from tensor
        name in the flatbuffer.
      description: human readable description of what the tensor does.

    Returns:
      The current Writer instance to allow chained operation.
    """
    assert class_names, 'class_name can not be empty'
    if locale_to_labels:
      for _, labels in locale_to_labels.items():
        assert len(class_names) == len(
            labels), 'all label files should have the same number of entries'

    # Task Lib uses the first label file for class_name
    label_files = [
        metadata_info.LabelFileMd(
            self._export_labels('labels.txt', class_names),
            # If no locale file attached, use class_name for display name.
            # 'en' is the default display name locale.
            locale=None if locale_to_labels else 'en')
    ]
    if locale_to_labels:
      for locale, labels in locale_to_labels.items():
        label_files.append(
            metadata_info.LabelFileMd(
                self._export_labels('labels_{}.txt'.format(locale), labels),
                locale=locale))

    calibration_md = None
    if score_transformation_type and score_calibration:
      calibration_md = metadata_info.ScoreCalibrationMd(
          score_transformation_type=score_transformation_type,
          default_score=default_calibrated_score or 0.,
          file_path=self._export_calibration_file('score_calibration.txt',
                                                  score_calibration))

    idx = len(self._output_mds)
    output_md = metadata_info.ClassificationTensorMd(
        name=name,
        description=description,
        label_files=label_files,
        tensor_type=self._output_tensor_type(idx),
        score_calibration_md=calibration_md,
    )
    self._output_mds.append(output_md)
    return self
