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
import numpy as np
from typing import List, Tuple, Optional

from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.processor.proto import segmentation_options_pb2
from tensorflow_lite_support.python.task.processor.proto import segmentations_pb2
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.task.vision.core.pybinds import image_utils
from tensorflow_lite_support.python.task.vision.pybinds import _pywrap_image_segmenter
from tensorflow_lite_support.python.task.vision.pybinds import image_segmenter_options_pb2

_ProtoOutputType = segmentation_options_pb2.OutputType
_ProtoImageSegmenterOptions = image_segmenter_options_pb2.ImageSegmenterOptions
_CppImageSegmenter = _pywrap_image_segmenter.ImageSegmenter
_BaseOptions = base_options_pb2.BaseOptions


@dataclasses.dataclass
class ColoredLabel:
  label: str
  """The label name."""

  color: Tuple[int, int, int]
  """The RGB representation of the label's color."""


@dataclasses.dataclass
class Segmentation:
  colored_labels: List[ColoredLabel]
  """The map between RGB color and label name."""

  masks: np.ndarray
  """The pixel mask representing the segmentation result."""

  output_type: _ProtoOutputType
  """The format of the model output."""

  width: int
  """Width of the mask."""

  height: int
  """Height of the mask."""


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
    base_options = _BaseOptions(file_name=file_path)
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
    # proto_options = _ProtoImageSegmenterOptions()
    # proto_options.base_options.CopyFrom(
    #     task_utils.ConvertToProtoBaseOptions(options.base_options))
    #
    # # Updates values from classification_options.
    # if options.segmentation_options:
    #   if options.segmentation_options.display_names_locale:
    #     proto_options.display_names_locale = options.segmentation_options.display_names_locale
    #   if options.segmentation_options.output_type:
    #     proto_options.output_type = options.segmentation_options.output_type
    #
    # segmenter = _CppImageSegmenter.create_from_options(proto_options)
    segmenter = _CppImageSegmenter.create_from_options(
        options.base_options, options.segmentation_options)
    return cls(options, segmenter)

  def segment(
      self,
      image: tensor_image.TensorImage
  ) -> Segmentation:
    """Performs segmentation on the provided TensorImage and postprocesses
    the segmentation result.

    Args:
      image: Tensor image, used to extract the feature vectors.
    Returns:
      segmentation output.
    Raises:
      status.StatusNotOk if failed to get the feature vector. Need to import the
        module to catch this error: `from pybind11_abseil
        import status`, see
        https://github.com/pybind/pybind11_abseil#abslstatusor.
    """
    image_data = image_utils.ImageData(image.buffer)
    segmentation_result = self._segmenter.segment(image_data)
    return self._postprocess(segmentation_result)

  def _postprocess(
      self,
      segmentation_result: segmentations_pb2.SegmentationResult
  ) -> Segmentation:
    """Post-process the output segmentation result into segmentation output.

    Args:
      segmentation_result: segmentation result, used for post-processing.
    Returns:
      segmentation output.
    """
    segmentation = segmentation_result.segmentation[0]
    output_type = self._options.segmentation_options.output_type

    if output_type == _ProtoOutputType.CATEGORY_MASK:
      masks = np.array(bytearray(segmentation.category_mask))

    elif output_type == _ProtoOutputType.CONFIDENCE_MASK:
      confidence_masks = segmentation.confidence_masks.confidence_mask
      masks = np.array([confidence_masks[index].value
                        for index in range(len(confidence_masks))])

    colored_labels = [
      ColoredLabel(colored_label.class_name,
                   (colored_label.r, colored_label.g, colored_label.b))
      for colored_label in segmentation.colored_labels]

    return Segmentation(
      colored_labels=colored_labels, masks=masks, output_type=output_type,
      width=segmentation.width, height=segmentation.height)
