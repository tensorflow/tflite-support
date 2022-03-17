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
"""Image segmenter task."""

import dataclasses
from typing import Optional

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.core import task_utils
from tensorflow_lite_support.python.task.processor.proto import segmentation_options_pb2
from tensorflow_lite_support.python.task.processor.proto import segmentations_pb2
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.task.vision.core.pybinds import image_utils
from tensorflow_lite_support.python.task.vision.pybinds import _pywrap_image_segmenter
from tensorflow_lite_support.python.task.vision.pybinds import image_segmenter_options_pb2

_ProtoImageSegmenterOptions = image_segmenter_options_pb2.ImageSegmenterOptions
_CppImageSegmenter = _pywrap_image_segmenter.ImageSegmenter
_BaseOptions = task_options.BaseOptions
_ExternalFile = task_options.ExternalFile


@dataclasses.dataclass
class ImageSegmenterOptions:
  """Options for the image segmenter task."""
  base_options: _BaseOptions
  segmentation_options: Optional[
      segmentation_options_pb2.SegmentationOptions] = None


class ImageSegmenter(object):
  """Class that performs segmentation on images."""

  def __init__(self, options: ImageSegmenterOptions,
               segmenter: _CppImageSegmenter) -> None:
    """Initializes the `ImageSegmenter` object."""
    # Creates the object of C++ ImageSegmenter class.
    self._options = options
    self._segmenter = segmenter

  @classmethod
  def create_from_file(cls, file_path: str) -> "ImageSegmenter":
    """Creates the `ImageSegmenter` object from a TensorFlow Lite model.

    Args:
      file_path: Path to the model.
    Returns:
      `ImageSegmenter` object that's created from `options`.
    Raises:
      status.StatusNotOk if failed to create `ImageSegmenter` object from the
      provided file such as invalid file.
    """
    # TODO(b/220931229): Raise RuntimeError instead of status.StatusNotOk.
    # Need to import the module to catch this error:
    # `from pybind11_abseil import status`
    # see https://github.com/pybind/pybind11_abseil#abslstatusor.
    base_options = _BaseOptions(
        model_file=_ExternalFile(file_name=file_path))
    options = ImageSegmenterOptions(base_options=base_options)
    return cls.create_from_options(options)

  @classmethod
  def create_from_options(cls,
                          options: ImageSegmenterOptions) -> "ImageSegmenter":
    """Creates the `ImageSegmenter` object from image segmenter options.

    Args:
      options: Options for the image segmenter task.
    Returns:
      `ImageSegmenter` object that's created from `options`.
    Raises:
      status.StatusNotOk if failed to create `ImageSegmenter` object from
      `ImageSegmenterOptionsn` such as missing the model.
    """
    # TODO(b/220931229): Raise RuntimeError instead of status.StatusNotOk.
    # Need to import the module to catch this error:
    # `from pybind11_abseil import status`
    # see https://github.com/pybind/pybind11_abseil#abslstatusor.
    proto_options = _ProtoImageSegmenterOptions()
    proto_options.base_options.CopyFrom(
        task_utils.ConvertToProtoBaseOptions(options.base_options))

    # Updates values from classification_options.
    if options.segmentation_options:
      if options.segmentation_options.display_names_locale:
        proto_options.display_names_locale = options.segmentation_options.display_names_locale
      if options.segmentation_options.output_type:
        proto_options.output_type = options.segmentation_options.output_type

    segmenter = _CppImageSegmenter.create_from_options(proto_options)

    return cls(options, segmenter)

  def segment(
      self,
      image: tensor_image.TensorImage
  ) -> segmentations_pb2.SegmentationResult:
    """Performs segmentation on the provided TensorImage.

    Args:
      image: Tensor image, used to extract the feature vectors.
    Returns:
      segmentation result.
    Raises:
      status.StatusNotOk if failed to get the feature vector. Need to import the
        module to catch this error: `from pybind11_abseil
        import status`, see
        https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    image_data = image_utils.ImageData(image.get_buffer())
    return self._segmenter.segment(image_data)
