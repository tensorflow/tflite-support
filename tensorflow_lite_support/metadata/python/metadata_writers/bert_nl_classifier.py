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
"""Writes metadata and label file to the Bert NL classifier models."""

from typing import List, Optional, Union

import flatbuffers
from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata.python import metadata as _metadata
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_info
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_writer
from tensorflow_lite_support.metadata.python.metadata_writers import writer_utils

_MODEL_NAME = "BertNLClassifier"
_MODEL_DESCRIPTION = ("Classify the input text into a set of known categories.")
_INPUT_IDS_NAME = "input_text"
_INPUT_IDS_DESCRIPTION = ("Embedding vectors representing the input text to be "
                          "classified.")
_INPUT_IDS_NAME = "ids"
_INPUT_IDS_DESCRIPTION = "Tokenized ids of the input text."
_INPUT_MASK_NAME = "mask"
_INPUT_MASK_DESCRIPTION = ("Mask with 1 for real tokens and 0 for padding "
                           "tokens.")
_INPUT_SEGMENT_IDS_NAME = "segment_ids"
_INPUT_SEGMENT_IDS_DESCRIPTION = (
    "0 for the first sequence, 1 for the second sequence if "
    "exists.")
_OUTPUT_NAME = "probability"
_OUTPUT_DESCRIPTION = "Probabilities of the labels respectively."


class MetadataWriter(metadata_writer.MetadataWriter):
  """Writes metadata into the NL classifier."""

  @classmethod
  def create_from_metadata_info(
      cls,
      model_buffer: bytearray,
      general_md: Optional[metadata_info.GeneralMd] = None,
      input_ids_md: Optional[metadata_info.TensorMd] = None,
      input_mask_md: Optional[metadata_info.TensorMd] = None,
      input_segment_ids_md: Optional[metadata_info.TensorMd] = None,
      tokenizer_md: Union[None, metadata_info.BertTokenizerMd,
                          metadata_info.SentencePieceTokenizerMd] = None,
      output_md: Optional[metadata_info.ClassificationTensorMd] = None):
    """Creates MetadataWriter based on general/input/output information.

    Args:
      model_buffer: valid buffer of the model file.
      general_md: general infromation about the model. If not specified, default
        general metadata will be generated.
      input_ids_md: input ids tensor informaton. The ids tensor represents the
        tokenized ids of input text.
      input_mask_md: input ids tensor informaton. The mask tensor represents the
        mask with 1 for real tokens and 0 for padding tokens.
      input_segment_ids_md: input ids tensor informaton. In the segment tesnor,
        `0` stands for the first sequence, and `1` stands for the second
        sequence if exists.
      tokenizer_md: information of the tokenizer in the input text tensor, if
        any. Supported tokenziers are: `BertTokenizer` [1] and
          `SentencePieceTokenizer` [2]. If the tokenizer is `RegexTokenizer`
          [3], refer to `nl_classifier.MetadataWriter`.
        [1]:
        https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L436
        [2]:
        https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L473
        [3]:
        https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L475
      output_md: output classification tensor informaton, if not specified,
        default output metadata will be generated.

    Returns:
      A MetadataWriter object.

    Raises:
      ValueError: if the type of tokenizer_md is unsupported.
    """
    if general_md is None:
      general_md = metadata_info.GeneralMd(
          name=_MODEL_NAME, description=_MODEL_DESCRIPTION)

    if input_ids_md is None:
      input_ids_md = metadata_info.TensorMd(
          name=_INPUT_IDS_NAME, description=_INPUT_IDS_DESCRIPTION)

    if input_mask_md is None:
      input_mask_md = metadata_info.TensorMd(
          name=_INPUT_MASK_NAME, description=_INPUT_MASK_DESCRIPTION)

    if input_segment_ids_md is None:
      input_segment_ids_md = metadata_info.TensorMd(
          name=_INPUT_SEGMENT_IDS_NAME,
          description=_INPUT_SEGMENT_IDS_DESCRIPTION)

    if output_md is None:
      output_md = metadata_info.ClassificationTensorMd(
          name=_OUTPUT_NAME, description=_OUTPUT_DESCRIPTION)

    if output_md.associated_files is None:
      output_md.associated_files = []

    # Create subgraph info.
    subgraph_metadata = _metadata_fb.SubGraphMetadataT()
    subgraph_metadata.inputTensorMetadata = [
        input_ids_md.create_metadata(),
        input_mask_md.create_metadata(),
        input_segment_ids_md.create_metadata()
    ]
    subgraph_metadata.outputTensorMetadata = [output_md.create_metadata()]

    # Create tokenzier metadata.
    if not isinstance(tokenizer_md, (type(None), metadata_info.BertTokenizerMd,
                                     metadata_info.SentencePieceTokenizerMd)):
      raise ValueError(
          "The type of tokenizer_options, {}, is unsupported".format(
              type(tokenizer_md)))

    tokenizer_files = []
    if tokenizer_md:
      tokenizer = tokenizer_md.create_metadata()
      subgraph_metadata.inputProcessUnits = [tokenizer]
      tokenizer_files = writer_utils.get_tokenizer_associated_files(
          tokenizer.options)

    # Create model metadata
    model_metadata = general_md.create_metadata()
    model_metadata.subgraphMetadata = [subgraph_metadata]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_metadata.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)

    return cls(
        model_buffer,
        metadata_buffer=b.Output(),
        associated_files=[
            file.file_path for file in output_md.associated_files
        ] + tokenizer_files)

  @classmethod
  def create_for_inference(
      cls, model_buffer: bytearray,
      tokenizer_md: Union[metadata_info.BertTokenizerMd,
                          metadata_info.SentencePieceTokenizerMd],
      label_file_paths: List[str]):
    """Creates mandatory metadata for TFLite Support inference.

    The parameters required in this method are mandatory when using TFLite
    Support features, such as Task library and Codegen tool (Android Studio ML
    Binding). Other metadata fields will be set to default. If other fields need
    to be filled, use the method `create_from_metadata_info` to edit them.

    Args:
      model_buffer: valid buffer of the model file.
      tokenizer_md: information of the tokenizer in the input text tensor, if
        any. Supported tokenziers are: `BertTokenizer` [1] and
          `SentencePieceTokenizer` [2]. If the tokenizer is `RegexTokenizer`
          [3], refer to `nl_classifier.MetadataWriter`.
        [1]:
        https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L436
        [2]:
        https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L473
        [3]:
        https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L475
      label_file_paths: paths to the label files [4] in the classification
        tensor. Pass in an empty list, If the model does not have any label
        file.
        [4]:
        https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L95

    Returns:
      A MetadataWriter object.
    """
    output_md = metadata_info.ClassificationTensorMd(
        name=_OUTPUT_NAME,
        description=_OUTPUT_DESCRIPTION,
        label_files=[
            metadata_info.LabelFileMd(file_path=file_path)
            for file_path in label_file_paths
        ],
        tensor_type=writer_utils.get_output_tensor_types(model_buffer)[0])

    return cls.create_from_metadata_info(
        model_buffer, tokenizer_md=tokenizer_md, output_md=output_md)
