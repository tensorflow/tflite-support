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
"""Image classifier task."""

import dataclasses
from typing import Optional

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.core.proto import configuration_pb2
from tensorflow_lite_support.python.task.processor import processor_options
from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2
from tensorflow_lite_support.python.task.processor.proto import classifications_pb2
from tensorflow_lite_support.python.task.processor.proto import image_classifier_options_pb2
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.task.vision.core.pybinds import image_utils
from tensorflow_lite_support.python.task.vision.pybinds import image_classifier as _image_classifier


@dataclasses.dataclass
class ImageClassifierOptions:
  """Options for the image classifier task."""
  base_options: task_options.BaseOptions
  classifier_options: Optional[processor_options.ClassificationOptions] = None


def _build_proto_options(
    options: ImageClassifierOptions
) -> image_classifier_options_pb2.ClassifierOptions:
  """Builds the protobuf image classifier options."""
  # Builds the initial proto_options.
  proto_options = image_classifier_options_pb2.ClassifierOptions()

  # Updates values from base_options.
  proto_options.model_file_with_metadata.file_name = options.base_options.model_file
  proto_options.num_threads = options.base_options.num_threads
  if options.base_options.use_coral:
    proto_options.compute_settings.tflite_settings.delegate = configuration_pb2.Delegate.EDGETPU_CORAL

  # Updates values from classifier_options.
  if options.classifier_options:
    if options.classifier_options.display_names_locale is not None:
      proto_options.display_names_locale = options.classifier_options.display_names_locale
    if options.classifier_options.max_results is not None:
      proto_options.max_results = options.classifier_options.max_results
    if options.classifier_options.score_threshold is not None:
      proto_options.score_threshold = options.classifier_options.score_threshold
    if options.classifier_options.label_allowlist is not None:
      proto_options.class_name_whitelist.extend(options.classifier_options.label_allowlist)
    if options.classifier_options.label_denylist is not None:
      proto_options.class_name_blacklist.extend(options.classifier_options.label_denylist)

  return proto_options


class ImageClassifier(object):
  """Class that performs classification on images."""

  def __init__(
      self,
      options: ImageClassifierOptions,
  ) -> None:
    """Initializes the `ImageClassifier` object.

    Args:
      options: Options for the image classifier task.

    Raises:
      status.StatusNotOk if failed to create `ImageClassifier` object from
        `ImageClassifierOptions` such as missing the model. Need to import the
        module to catch this error: `from pybind11_abseil import status`, see
        https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    self._options = options

    # Creates the object of C++ ImageClassifier class.
    proto_options = _build_proto_options(options)
    self._classifier = _image_classifier.ImageClassifier.create_from_options(
        proto_options)

  @classmethod
  def create_from_options(cls,
                          options: ImageClassifierOptions) -> "ImageClassifier":
    """Creates the `ImageClassifier` object from image classifier options.

    Args:
      options: Options for the image classifier task.

    Returns:
      `ImageClassifier` object that's created from `options`.

    Raises:
      status.StatusNotOk if failed to create `ImageClassifier` object from
        `ImageClassifierOptions` such as missing the model. Need to import the
        module to catch this error: `from pybind11_abseil import status`, see
        https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    return cls(options)

  def classify(
      self,
      image: tensor_image.TensorImage,
      bounding_box: Optional[bounding_box_pb2.BoundingBox] = None
  ) -> classifications_pb2.ClassificationResult:
    """Performs classification on the provided TensorImage.

    Args:
      image: Tensor image, used to extract the feature vectors.
      bounding_box: Bounding box, optional. If set, performed feature vector
        extraction only on the provided region of interest. Note that the region
        of interest is not clamped, so this method will fail if the region is
        out of bounds of the input image.

    Returns:
      classification result.

    Raises:
      status.StatusNotOk if failed to get the feature vector. Need to import
        the module to catch this error: `from pybind11_abseil import status`,
        see https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    image_data = image_utils.ImageData(image.get_buffer())
    if bounding_box is None:
      return self._classifier.classify(image_data)

    return self._classifier.classify(image_data, bounding_box)
