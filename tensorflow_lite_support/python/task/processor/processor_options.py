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
"""Options to configure Task APIs."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Category:
  """Category class."""

  # The index of the class in the corresponding label map, usually packed in
  # the TFLite Model Metadata [1].
  #
  # [1]: https:#www.tensorflow.org/lite/convert/metadata
  index: Optional[int] = None

  # The score for this class e.g. (but not necessarily) a probability in [0,1].
  score: Optional[float] = None

  # A human readable name of the class filled from the label map.
  display_name: Optional[str] = None

  # An ID for the class, not necessarily human-readable (e.g. a Google
  # Knowledge Graph ID [1]), filled from the label map.
  #
  # [1]: https://developers.google.com/knowledge-graph
  class_name: Optional[str] = None


@dataclass
class Classifications:
  """Classifications class"""

  # The array of predicted categories, usually sorted by descending scores (e.g.
  # from high to low probability).
  categories: List[Category] = field(default_factory=list)

  # The index of the classifier head these classes refer to. This is useful for
  # multi-head models.
  head_index: Optional[int] = None

  # The name of the classifier head, which is the corresponding tensor metadata
  # name. See
  # https://github.com/tensorflow/tflite-support/blob/710e323265bfb71fdbdd72b3516e00cff15c0326/tensorflow_lite_support/metadata/metadata_schema.fbs#L545
  head_name: Optional[str] = None


@dataclass
class ClassificationResult:
  """ClassificationResult class"""

  classifications: List[Classifications] = field(default_factory=list)


@dataclass
class NLClassificationOptions:
  """Options for NL classifier processor.
   Configure the input/output tensors for NLClassifier:

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
  """

  # Name of the input text tensor, if the model has multiple inputs. Only the
  # input tensor specified will be used for inference; other input tensors will
  # be ignored. Default to "INPUT".
  #
  # See the "Configure the input/output tensors for NLClassifier" section above
  # for more details.
  input_tensor_name: Optional[str] = "INPUT"

  # Name of the output score tensor, if the model has multiple outputs. Default
  # to "OUTPUT_SCORE".
  #
  # See the "Configure the input/output tensors for NLClassifier" section above
  # for more details.
  output_score_tensor_name: Optional[str] = "OUTPUT_SCORE"

  # Name of the output label tensor, if the model has multiple outputs. Default
  # to "OUTPUT_LABEL".
  #
  # See the "Configure the input/output tensors for NLClassifier" section above
  # for more details.
  #
  # By default, label file should be packed with
  # the output score tensor through Model Metadata. See the MetadataWriter for
  # NLClassifier [1]. NLClassifier reads and parses labels from the label
  # file automatically. However, some models may output a specific label tensor
  # instead. In this case, NLClassifier reads labels from the output label
  # tensor.
  #
  # [1]: https://www.tensorflow.org/lite/convert/metadata_writer_tutorial#natural_language_classifiers
  output_label_tensor_name: Optional[str] = "OUTPUT_LABEL"

  # Index of the input text tensor among all input tensors, if the model has
  # multiple inputs. Only the input tensor specified will be used for
  # inference; other input tensors will be ignored. Default to 0.
  #
  # See the "Configure the input/output tensors for NLClassifier" section above
  # for more details.
  input_tensor_index: Optional[int] = 0

  # Index of the output score tensor among all output tensors, if the model has
  # multiple outputs. Default to 0.
  #
  # See the "Configure the input/output tensors for NLClassifier" section above
  # for more details.
  output_score_tensor_index: Optional[int] = 0

  # Index of the optional output label tensor among all output tensors, if the
  # model has multiple outputs.
  #
  # See the comment above `output_label_tensor_name` for more information about
  # what the output label tensor is.
  #
  # See the "Configure the input/output tensors for NLClassifier" section above
  # for more details.
  #
  # `output_label_tensor_index` defaults to -1, meaning to disable searching
  # the output label tensor as it might be optional.
  output_label_tensor_index: Optional[int] = -1
