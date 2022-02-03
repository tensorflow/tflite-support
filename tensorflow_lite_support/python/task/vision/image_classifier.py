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
from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2
from tensorflow_lite_support.python.task.processor.proto import classification_options_pb2
from tensorflow_lite_support.python.task.core.proto import classifications_pb2
from tensorflow_lite_support.python.task.core.utils import build_proto_base_options
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.task.vision.core.pybinds import image_utils
from tensorflow_lite_support.python.task.vision.pybinds import _pywrap_image_classifier
from tensorflow_lite_support.python.task.vision.pybinds import image_classifier_options_pb2

_ProtoImageClassifierOptions = image_classifier_options_pb2.ImageClassifierOptions
_CppImageClassifier = _pywrap_image_classifier.ImageClassifier


@dataclasses.dataclass
class ImageClassifierOptions:
    """Options for the image embedder task."""
    base_options: task_options.BaseOptions
    embedding_options: Optional[classification_options_pb2.ClassificationOptions] = None


def _build_proto_options(
        options: ImageClassifierOptions) -> _ProtoImageClassifierOptions:
    """Builds the protobuf image embdder options."""
    # Builds the initial proto_options.
    proto_options = _ProtoImageClassifierOptions()

    # Updates values from base_options.
    proto_options = build_proto_base_options(proto_options, options)

    # Updates values from classifier_options.
    if options.classification_options:
        if options.classification_options.display_names_locale is not None:
            proto_options.display_names_locale = options.classification_options.display_names_locale
        if options.classification_options.max_results is not None:
            proto_options.max_results = options.classification_options.max_results
        if options.classification_options.score_threshold is not None:
            proto_options.score_threshold = options.classification_options.score_threshold
        if options.classification_options.class_name_allowlist is not None:
            proto_options.class_name_whitelist.extend(options.classification_options.class_name_allowlist)
        if options.classification_options.class_name_denylist is not None:
            proto_options.class_name_blacklist.extend(options.classification_options.class_name_denylist)

    return proto_options


class ImageClassifier(object):
    """Class that performs classification on images."""

    def __init__(self, classifier: _CppImageClassifier) -> None:
        """Initializes the `ImageClassifier` object."""
        self._classifier = classifier

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
        # Creates the object of C++ ImageClassifier class.
        proto_options = _build_proto_options(options)
        classifier = _CppImageClassifier.create_from_options(proto_options)

        return cls(classifier)

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
