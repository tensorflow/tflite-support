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
"""Tests for image_segmenter."""

import enum
import json

import numpy as np
from absl.testing import parameterized
from google.protobuf import json_format
# TODO(b/220067158): Change to import tensorflow and leverage tf.test once
# fixed the dependency issue.
import unittest
from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.processor.proto import segmentations_pb2
from tensorflow_lite_support.python.task.processor.proto import segmentation_options_pb2
from tensorflow_lite_support.python.task.vision import image_segmenter
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.test import base_test
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_pb2.BaseOptions
_OutputType = segmentation_options_pb2.OutputType
_ImageSegmenter = image_segmenter.ImageSegmenter
_ImageSegmenterOptions = image_segmenter.ImageSegmenterOptions

_MODEL_FILE = 'deeplabv3.tflite'
_IMAGE_FILE = 'segmentation_input_rotation0.jpg'
_SEGMENTATION_FILE = 'segmentation_golden_rotation0.png'
_EXPECTED_COLORED_LABELS = [
  {'r': 0, 'g': 0, 'b': 0, 'className': 'background'},
  {'r': 128, 'g': 0, 'b': 0, 'className': 'aeroplane'},
  {'r': 0, 'g': 128, 'b': 0, 'className': 'bicycle'},
  {'r': 128, 'g': 128, 'b': 0, 'className': 'bird'},
  {'r': 0, 'g': 0, 'b': 128, 'className': 'boat'},
  {'r': 128, 'g': 0, 'b': 128, 'className': 'bottle'},
  {'r': 0, 'g': 128, 'b': 128, 'className': 'bus'},
  {'r': 128, 'g': 128, 'b': 128, 'className': 'car'},
  {'r': 64, 'g': 0, 'b': 0, 'className': 'cat'},
  {'r': 192, 'g': 0, 'b': 0, 'className': 'chair'},
  {'r': 64, 'g': 128, 'b': 0, 'className': 'cow'},
  {'r': 192, 'g': 128, 'b': 0, 'className': 'dining table'},
  {'r': 64, 'g': 0, 'b': 128, 'className': 'dog'},
  {'r': 192, 'g': 0, 'b': 128, 'className': 'horse'},
  {'r': 64, 'g': 128, 'b': 128, 'className': 'motorbike'},
  {'r': 192, 'g': 128, 'b': 128, 'className': 'person'},
  {'r': 0, 'g': 64, 'b': 0, 'className': 'potted plant'},
  {'r': 128, 'g': 64, 'b': 0, 'className': 'sheep'},
  {'r': 0, 'g': 192, 'b': 0, 'className': 'sofa'},
  {'r': 128, 'g': 192, 'b': 0, 'className': 'train'},
  {'r': 0, 'g': 64, 'b': 128, 'className': 'tv'}
]
_MASK_MAGNIFICATION_FACTOR = 10
_MATCH_PIXELS_THRESHOLD = 0.01
_ACCEPTABLE_ERROR_RANGE = 0.000001


def _create_segmenter_from_options(base_options, **segmentation_options):
  segmentation_options = segmentation_options_pb2.SegmentationOptions(
    **segmentation_options)
  options = _ImageSegmenterOptions(
    base_options=base_options,
    segmentation_options=segmentation_options)
  segmenter = _ImageSegmenter.create_from_options(options)
  return segmenter


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageSegmenterTest(parameterized.TestCase, base_test.BaseTestCase):

  def setUp(self):
    super().setUp()
    self.test_image_path = test_util.get_test_data_path(_IMAGE_FILE)
    self.test_seg_path = test_util.get_test_data_path(_SEGMENTATION_FILE)
    self.model_path = test_util.get_test_data_path(_MODEL_FILE)

  @parameterized.parameters(
    (ModelFileType.FILE_NAME, _EXPECTED_COLORED_LABELS),
    (ModelFileType.FILE_CONTENT, _EXPECTED_COLORED_LABELS))
  def test_segment_model(self, model_file_type, expected_colored_labels):
    # Creates segmenter.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, "rb") as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError("model_file_type is invalid.")

    segmenter = _create_segmenter_from_options(base_options)

    # Loads image.
    image = tensor_image.TensorImage.create_from_file(self.test_image_path)

    # Performs image segmentation on the input.
    segmentation = segmenter.segment(image).segmentation[0]
    segmentation_colored_labels = json.loads(
      json_format.MessageToJson(segmentation))['coloredLabels']

    # Comparing results.
    self.assertDeepAlmostEqual(
      segmentation_colored_labels, expected_colored_labels)

  def test_segmentation_category_mask(self):
    """Check if category mask match with ground truth."""
    # Creates segmenter.
    base_options = _BaseOptions(file_name=self.model_path)
    segmenter = _create_segmenter_from_options(
      base_options, output_type=_OutputType.CATEGORY_MASK)

    # Loads image.
    image = tensor_image.TensorImage.create_from_file(self.test_image_path)

    # Performs image segmentation on the input.
    segmentation = segmenter.segment(image).segmentation[0]
    result_pixels = np.array(bytearray(segmentation.category_mask))

    # Loads ground truth segmentation file.
    gt_segmentation = tensor_image.TensorImage.create_from_file(
      self.test_seg_path)
    gt_segmentation_array = gt_segmentation.buffer
    gt_segmentation_shape = gt_segmentation_array.shape
    num_pixels = gt_segmentation_shape[0] * gt_segmentation_shape[1]
    ground_truth_pixels = gt_segmentation_array.flatten()

    inconsistent_pixels = 0

    for index in range(num_pixels):
      inconsistent_pixels += (
        result_pixels[index] * _MASK_MAGNIFICATION_FACTOR !=
        ground_truth_pixels[index])

    self.assertLessEqual(inconsistent_pixels / num_pixels,
                         _MATCH_PIXELS_THRESHOLD,
      "Segmentation mask value must be the same size as ground truth.")

  def test_segmentation_confidence_mask(self):
    """Check if top-left corner has expected confidences and also verify if the
     confidence mask matches with the category mask."""
    # Create BaseOptions from model file.
    base_options = _BaseOptions(file_name=self.model_path)

    # Loads image.
    image = tensor_image.TensorImage.create_from_file(self.test_image_path)

    # Run segmentation on the model in CATEGORY_MASK mode.
    segmenter = _create_segmenter_from_options(
      base_options, output_type=_OutputType.CATEGORY_MASK)

    # Performs image segmentation on the input and gets the category mask.
    segmentation = segmenter.segment(image).segmentation[0]
    category_mask = np.array(bytearray(segmentation.category_mask))

    # Run segmentation on the model in CATEGORY_MASK mode.
    segmenter = _create_segmenter_from_options(
      base_options, output_type=_OutputType.CONFIDENCE_MASK)

    # Performs image segmentation on the input again.
    segmentation = segmenter.segment(image).segmentation[0]

    # Gets the list of confidence masks and colored_labels.
    confidence_masks = segmentation.confidence_masks.confidence_mask
    colored_labels = segmentation.colored_labels

    # Check if confidence mask shape is correct.
    self.assertEqual(len(confidence_masks), len(colored_labels),
                     'Number of confidence masks must match with number of '
                     'categories.')

    # Gather the confidence masks in a single array `confidence_mask_array`.
    confidence_mask_array = np.array([confidence_masks[index].value
                             for index in range(len(confidence_masks))])

    # Compute the category mask from the created confidence mask.
    calculated_category_mask = np.argmax(confidence_mask_array, axis=0)
    self.assertListEqual(calculated_category_mask.tolist(),
                         category_mask.tolist())


if __name__ == '__main__':
  unittest.main()
