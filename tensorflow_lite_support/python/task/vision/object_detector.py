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
"""Object detector task."""

import dataclasses
from typing import Optional

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.core import task_utils
from tensorflow_lite_support.python.task.processor.proto import detection_options_pb2
from tensorflow_lite_support.python.task.processor.proto import detections_pb2
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.task.vision.core.pybinds import image_utils
from tensorflow_lite_support.python.task.vision.pybinds import _pywrap_object_detector
from tensorflow_lite_support.python.task.vision.pybinds import object_detector_options_pb2

_ProtoObjectDetectorOptions = object_detector_options_pb2.ObjectDetectorOptions
_CppObjectDetector = _pywrap_object_detector.ObjectDetector


@dataclasses.dataclass
class ObjectDetectorOptions:
  """Options for the object detector task."""
  base_options: task_options.BaseOptions
  detection_options: Optional[
    detection_options_pb2.DetectionOptions] = None


class ObjectDetector(object):
  """Class that performs object detection on images."""

  def __init__(self, detector: _CppObjectDetector) -> None:
    """Initializes the `ObjectDetector` object."""
    self._detector = detector

  @classmethod
  def create_from_options(cls,
                          options: ObjectDetectorOptions) -> "ObjectDetector":
    """Creates the `ObjectDetector` object from object detector options.

    Args:
      options: Options for the object detector task.
    Returns:
      `ObjectDetector` object that's created from `options`.
    Raises:
      status.StatusNotOk if failed to create `ObjectDetector` object from
        `ObjectDetectorOptions` such as missing the model. Need to import the
        module to catch this error: `from pybind11_abseil
        import status`, see
        https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    # Creates the object of C++ ObjectDetector class.
    proto_options = _ProtoObjectDetectorOptions()
    proto_options.base_options.CopyFrom(
        task_utils.ConvertToProtoBaseOptions(options.base_options))

    # Updates values from detection_options.
    if options.detection_options:
      if options.detection_options.display_names_locale:
        proto_options.display_names_locale = options.detection_options.display_names_locale
      if options.detection_options.max_results:
        proto_options.max_results = options.detection_options.max_results
      if options.detection_options.score_threshold:
        proto_options.score_threshold = options.detection_options.score_threshold
      if options.detection_options.class_name_allowlist:
        proto_options.class_name_whitelist.extend(
            options.detection_options.class_name_allowlist)
      if options.detection_options.class_name_denylist:
        proto_options.class_name_blacklist.extend(
            options.detection_options.class_name_denylist)

    detector = _CppObjectDetector.create_from_options(proto_options)

    return cls(detector)

  def detect(
      self,
      image: tensor_image.TensorImage
  ) -> detections_pb2.DetectionResult:
    """Performs object detection on the provided TensorImage.

    Args:
      image: Tensor image, used to extract the feature vectors.
    Returns:
      detection result.
    Raises:
      status.StatusNotOk if failed to get the feature vector. Need to import the
        module to catch this error: `from pybind11_abseil
        import status`, see
        https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    image_data = image_utils.ImageData(image.get_buffer())

    return self._detector.detect(image_data)
