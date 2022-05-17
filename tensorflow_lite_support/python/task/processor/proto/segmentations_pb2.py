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
"""Segmentations protobuf."""

import dataclasses
from typing import Any, Tuple, List

from tensorflow.tools.docs import doc_controls
from tensorflow_lite_support.cc.task.vision.proto import segmentations_pb2

_Segmentation = segmentations_pb2.Segmentation
_ConfidenceMask = segmentations_pb2.Segmentation.ConfidenceMask
_ColoredLabel = segmentations_pb2.Segmentation.ColoredLabel
_SegmentationResult = segmentations_pb2.SegmentationResult


@dataclasses.dataclass
class ConfidenceMask:
  """This is a flattened 2D-array in row major order. For each pixel, the
    value indicates the prediction confidence usually in the [0, 1] range
    where higher values represent a stronger confidence.
    Ultimately this is model specific, and other range of values might be
    used.

  Attributes:
    value: The value indicates the prediction confidence usually in the
      range [0, 1].
  """
  value: float

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ConfidenceMask:
    """Generates a protobuf object to pass to the C++ layer."""
    return _ConfidenceMask(value=self.value)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _ConfidenceMask) -> "ConfidenceMask":
    """Creates a `ConfidenceMask` object from the given protobuf object."""
    return ConfidenceMask(value=pb2_obj.value)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, ConfidenceMask):
      return False


@dataclasses.dataclass
class ColoredLabel:
  """Defines a label associated with an RGB color, for display purposes.

  Attributes:
    color: The RGB color components for the label, in the [0, 255] range.
    category_name: The class name, as provided in the label map packed in the
      TFLite ModelMetadata.
    display_name: The display name, as provided in the label map
      (if available) packed in the TFLite Model Metadata .
  """

  color: Tuple[int, int, int]
  category_name: str
  display_name: str

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ColoredLabel:
    """Generates a protobuf object to pass to the C++ layer."""
    r, g, b = self.color
    return _ColoredLabel(r=r, g=g, b=b,
                         class_name=self.category_name,
                         display_name=self.display_name)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _ColoredLabel) -> "ColoredLabel":
    """Creates a `ColoredLabel` object from the given protobuf object."""
    return ColoredLabel(color=(pb2_obj.r, pb2_obj.g, pb2_obj.b),
                        category_name=pb2_obj.class_name,
                        display_name=pb2_obj.display_name)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, ColoredLabel):
      return False


@dataclasses.dataclass
class Segmentation:
  """Represents one Segmentation object in the image segmenter's results.

  Attributes:
    colored_labels: A list of `ColoredLabel` objects.
    category_mask: A bytearray of the category mask.
    confidence_masks: A list of `ConfidenceMask` objects.
  """

  colored_labels: List[ColoredLabel]
  category_mask: bytearray = None
  confidence_masks: List[ConfidenceMask] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _Segmentation:
    """Generates a protobuf object to pass to the C++ layer."""

    if self.category_mask is not None:
      return _Segmentation(
        category_mask=bytes(self.category_mask),
        colored_labels=[
          colored_label.to_pb2() for colored_label in self.colored_labels])

    elif self.confidence_masks is not None:
      return _Segmentation(
        confidence_masks=[
          confidence_mask.to_pb2()
          for confidence_mask in self.confidence_masks],
        colored_labels=[
          colored_label.to_pb2()
          for colored_label in self.colored_labels])

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls,
      pb2_obj: _Segmentation
  ) -> "Segmentation":
    """Creates a `Segmentation` object from the given protobuf object."""

    if pb2_obj.category_mask:
      return Segmentation(
        category_mask=bytearray(pb2_obj.category_mask),
        colored_labels=[
          ColoredLabel.create_from_pb2(colored_label)
          for colored_label in pb2_obj.colored_labels])

    elif pb2_obj.confidence_masks.confidence_mask:
      return Segmentation(
        confidence_masks=[
          ConfidenceMask.create_from_pb2(
            pb2_obj.confidence_masks.confidence_mask[index])
          for index in range(len(pb2_obj.confidence_masks.confidence_mask))],
        colored_labels=[
          ColoredLabel.create_from_pb2(colored_label)
          for colored_label in pb2_obj.colored_labels])

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Segmentation):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class SegmentationResult:
  """Results of performing image segmentation.
  Note that at the time, a single `Segmentation` element is expected to be
  returned; the field is made repeated for later extension to e.g. instance
  segmentation models, which may return one segmentation per object.

  Attributes:
    segmentations: A list of `Segmentation` objects.
  """

  segmentations: List[Segmentation]

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _SegmentationResult:
    """Generates a protobuf object to pass to the C++ layer."""
    return _SegmentationResult(
      segmentation=[
        segmentation.to_pb2()
        for segmentation in self.segmentations])

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls,
      pb2_obj: _SegmentationResult
  ) -> "SegmentationResult":
    """Creates a `SegmentationResult` object from the given protobuf object."""
    return SegmentationResult(
      segmentations=[
        Segmentation.create_from_pb2(segmentation)
        for segmentation in pb2_obj.segmentation])

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, SegmentationResult):
      return False
