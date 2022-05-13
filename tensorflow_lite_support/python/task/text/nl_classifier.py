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
"""NL Classifier task."""

import dataclasses
from typing import Optional

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.core import task_utils
from tensorflow_lite_support.python.task.processor import processor_options
from tensorflow_lite_support.python.task.text.pybinds import _pywrap_nl_classifier
from tensorflow_lite_support.python.task.text.pybinds import nl_classifier_options_pb2

_ProtoNLClassifierOptions = nl_classifier_options_pb2.NLClassifierOptions
_CppNLClassifier = _pywrap_nl_classifier.NLClassifier
_NLClassificationOptions = processor_options.NLClassificationOptions
_Category = processor_options.Category
_Classifications = processor_options.Classifications
_ClassificationResult = processor_options.ClassificationResult


@dataclasses.dataclass
class NLClassifierOptions:
  """Options for the NL classifier task."""
  base_options: task_options.BaseOptions
  nl_classification_options: Optional[_NLClassificationOptions] = None


class NLClassifier(object):
  """Class that performs NL classification on text."""

  def __init__(self, options: NLClassifierOptions,
               cpp_classifier: _CppNLClassifier) -> None:
    """Initializes the `NLClassifier` object."""
    # Creates the object of C++ NLClassifier class.
    self._options = options
    self._classifier = cpp_classifier

  @classmethod
  def create_from_file(cls, file_path: str) -> "NLClassifier":
    """Creates the `NLClassifier` object from a TensorFlow Lite model.

    Args:
      file_path: Path to the model.

    Returns:
      `NLClassifier` object that's created from the model file.
    Raises:
      ValueError: If failed to create `NLClassifier` object from the provided
        file such as invalid file.
      RuntimeError: If other types of error occurred.
    """
    model_file = task_options.ExternalFile(file_name=file_path)
    base_options = task_options.BaseOptions(model_file=model_file)
    options = NLClassifierOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls, options: NLClassifierOptions) -> "NLClassifier":
    """Creates the `NLClassifier` object from NL classifier options.

    Args:
      options: Options for the NL classifier task.

    Returns:
      `NLClassifier` object that's created from `options`.
    Raises:
      ValueError: If failed to create `NLClassifier` object from
        `NLClassifierOptions` such as missing the model.
      RuntimeError: If other types of error occurred.
    """
    proto_options = _ProtoNLClassifierOptions()
    proto_options.base_options.CopyFrom(
      task_utils.ConvertToProtoBaseOptions(options.base_options))

    if options.nl_classification_options:
      if options.nl_classification_options.input_tensor_name is not None:
        proto_options.input_tensor_name = \
          options.nl_classification_options.input_tensor_name
      if options.nl_classification_options.output_score_tensor_name is not None:
        proto_options.output_score_tensor_name = \
          options.nl_classification_options.output_score_tensor_name
      if options.nl_classification_options.output_label_tensor_name is not None:
        proto_options.output_label_tensor_name = \
          options.nl_classification_options.output_label_tensor_name
      if options.nl_classification_options.input_tensor_index is not None:
        proto_options.input_tensor_index = \
          options.nl_classification_options.input_tensor_index
      if options.nl_classification_options.output_score_tensor_index is not None:
        proto_options.output_score_tensor_index = \
          options.nl_classification_options.output_score_tensor_index
      if options.nl_classification_options.output_label_tensor_index is not None:
        proto_options.output_label_tensor_index = \
          options.nl_classification_options.output_label_tensor_index

    classifier = _CppNLClassifier.create_from_options(proto_options)
    return cls(options, classifier)

  def classify(self, text: str) -> _ClassificationResult:
    """Performs actual NL classification on the provided text.

    Args:
      text: the input text, used to extract the feature vectors.

    Returns:
      classification result.

    Raises:
      ValueError: If any of the input arguments is invalid.
      RuntimeError: If failed to calculate the embedding vector.
    """
    classification_result = _ClassificationResult()
    proto_classification_result = self._classifier.classify(text)

    if proto_classification_result.classifications:
      for proto_classification in proto_classification_result.classifications:
        classifications = _Classifications(
          head_index=proto_classification.head_index)
        if proto_classification.classes:
          for proto_category in proto_classification.classes:
            category = _Category(score=proto_category.score,
                                 class_name=proto_category.class_name,
                                 display_name=proto_category.display_name)
            classifications.categories.append(category)
        classification_result.classifications.append(classifications)

    return classification_result

  @property
  def options(self) -> NLClassifierOptions:
    return self._options
