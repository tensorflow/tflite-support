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
"""NL Classification options protobuf."""

import dataclasses
from typing import Any, Optional

from tensorflow_lite_support.cc.task.processor.proto import nl_classification_options_pb2
from tensorflow_lite_support.python.task.core.optional_dependencies import doc_controls

_NLClassificationOptionsProto = nl_classification_options_pb2.NLClassificationOptions


@dataclasses.dataclass
class NLClassificationOptions:
  """Configure the input/output tensors for NL classification processor.

  - No special configuration is needed if the model has only one input tensor
  and one output tensor.

  - When the model has multiple input or output tensors, use the following
  configurations to specifiy the desired tensors:
    -- tensor names: `input_tensor_name`, `output_score_tensor_name`,
    `output_label_tensor_name`
    -- tensor indices: `input_tensor_index`, `output_score_tensor_index`,
    `output_label_tensor_index`
  Tensor names has higher priorities than tensor indices in locating the
  tensors. It means the tensors will be first located according to tensor
  names. If not found, then the tensors will be located according to tensor
  indices.

  - Failing to match the input text tensor or output score tensor with
  neither tensor names nor tensor indices will trigger a runtime error.
  However, failing to locate the output label tensor will not trigger an
  error because the label tensor is optional.

  Attributes:
    input_tensor_name: Name of the input text tensor, if the model has
      multiple inputs. Only the input tensor specified will be used for
      inference; other input tensors will be ignored. Defaults to "INPUT".
    output_score_tensor_name: Name of the output score tensor, if the model has
      multiple outputs. Defaults to "OUTPUT_SCORE".
    output_label_tensor_name: Name of the output label tensor, if the model has
      multiple outputs. Defaults to "OUTPUT_LABEL". By default, label file
      should be packed with the output score tensor through Model Metadata. See
      the MetadataWriter for NLClassifier [1]. NLClassifier reads and parses
      labels from the label file automatically. However, some models may output
      a specific label tensor instead. In this case, NLClassifier reads labels
      from the output label tensor.
    input_tensor_index: Index of the input text tensor among all input tensors,
      if the model has multiple inputs. Only the input tensor specified will be
      used for inference; other input tensors will be ignored. Defaults to 0.
    output_score_tensor_index: Index of the output score tensor among all
      output tensors, if the model has multiple outputs. Defaults to 0.
    output_label_tensor_index: Index of the optional output label tensor among
      all output tensors, if the model has multiple outputs.
      See `output_label_tensor_name` for more information about what the output
      label tensor is. Defaults to -1, meaning to disable searching the output
      label tensor as it might be optional.
  """

  input_tensor_name: Optional[str] = "INPUT"
  output_score_tensor_name: Optional[str] = "OUTPUT_SCORE"
  output_label_tensor_name: Optional[str] = "OUTPUT_LABEL"
  input_tensor_index: Optional[int] = 0
  output_score_tensor_index: Optional[int] = 0
  output_label_tensor_index: Optional[int] = -1

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _NLClassificationOptionsProto:
    """Generates a protobuf object to pass to the C++ layer."""
    return _NLClassificationOptionsProto(
      input_tensor_name=self.input_tensor_name,
      output_score_tensor_name=self.output_score_tensor_name,
      output_label_tensor_name=self.output_label_tensor_name,
      input_tensor_index=self.input_tensor_index,
      output_score_tensor_index=self.output_score_tensor_index,
      output_label_tensor_index=self.output_label_tensor_index)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls, pb2_obj: _NLClassificationOptionsProto) -> "NLClassificationOptions":
    """Creates a `NLClassificationOptions` object from the given protobuf object."""
    return NLClassificationOptions(
        input_tensor_name=pb2_obj.input_tensor_name,
        output_score_tensor_name=pb2_obj.output_score_tensor_name,
        output_label_tensor_name=pb2_obj.output_label_tensor_name,
        input_tensor_index=pb2_obj.input_tensor_index,
        output_score_tensor_index=pb2_obj.output_score_tensor_index,
        output_label_tensor_index=pb2_obj.output_label_tensor_index)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, NLClassificationOptions):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
