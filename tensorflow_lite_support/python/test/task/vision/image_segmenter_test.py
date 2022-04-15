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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.processor.proto import segmentation_options_pb2
from tensorflow_lite_support.python.task.vision import image_segmenter
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_pb2.BaseOptions
_OutputType = segmentation_options_pb2.SegmentationOptions.OutputType
_ImageSegmenter = image_segmenter.ImageSegmenter
_ImageSegmenterOptions = image_segmenter.ImageSegmenterOptions

_MODEL_FILE = 'deeplabv3.tflite'
_IMAGE_FILE = 'segmentation_input_rotation0.jpg'
_SEGMENTATION_FILE = 'segmentation_golden_rotation0.png'
_EXPECTED_COLORED_LABELS = [{
    'r': 0,
    'g': 0,
    'b': 0,
    'class_name': 'background'
}, {
    'r': 128,
    'g': 0,
    'b': 0,
    'class_name': 'aeroplane'
}, {
    'r': 0,
    'g': 128,
    'b': 0,
    'class_name': 'bicycle'
}, {
    'r': 128,
    'g': 128,
    'b': 0,
    'class_name': 'bird'
}, {
    'r': 0,
    'g': 0,
    'b': 128,
    'class_name': 'boat'
}, {
    'r': 128,
    'g': 0,
    'b': 128,
    'class_name': 'bottle'
}, {
    'r': 0,
    'g': 128,
    'b': 128,
    'class_name': 'bus'
}, {
    'r': 128,
    'g': 128,
    'b': 128,
    'class_name': 'car'
}, {
    'r': 64,
    'g': 0,
    'b': 0,
    'class_name': 'cat'
}, {
    'r': 192,
    'g': 0,
    'b': 0,
    'class_name': 'chair'
}, {
    'r': 64,
    'g': 128,
    'b': 0,
    'class_name': 'cow'
}, {
    'r': 192,
    'g': 128,
    'b': 0,
    'class_name': 'dining table'
}, {
    'r': 64,
    'g': 0,
    'b': 128,
    'class_name': 'dog'
}, {
    'r': 192,
    'g': 0,
    'b': 128,
    'class_name': 'horse'
}, {
    'r': 64,
    'g': 128,
    'b': 128,
    'class_name': 'motorbike'
}, {
    'r': 192,
    'g': 128,
    'b': 128,
    'class_name': 'person'
}, {
    'r': 0,
    'g': 64,
    'b': 0,
    'class_name': 'potted plant'
}, {
    'r': 128,
    'g': 64,
    'b': 0,
    'class_name': 'sheep'
}, {
    'r': 0,
    'g': 192,
    'b': 0,
    'class_name': 'sofa'
}, {
    'r': 128,
    'g': 192,
    'b': 0,
    'class_name': 'train'
}, {
    'r': 0,
    'g': 64,
    'b': 128,
    'class_name': 'tv'
}]
_MASK_MAGNIFICATION_FACTOR = 10
_MATCH_PIXELS_THRESHOLD = 0.01


def _create_segmenter_from_options(base_options, **segmentation_options):
  segmentation_options = segmentation_options_pb2.SegmentationOptions(
      **segmentation_options)
  options = _ImageSegmenterOptions(
      base_options=base_options, segmentation_options=segmentation_options)
  segmenter = _ImageSegmenter.create_from_options(options)
  return segmenter


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class ImageSegmenterTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.test_image_path = test_util.get_test_data_path(_IMAGE_FILE)
    self.test_seg_path = test_util.get_test_data_path(_SEGMENTATION_FILE)
    self.model_path = test_util.get_test_data_path(_MODEL_FILE)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    segmenter = _ImageSegmenter.create_from_file(self.model_path)
    self.assertIsInstance(segmenter, _ImageSegmenter)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(file_name=self.model_path)
    options = _ImageSegmenterOptions(base_options=base_options)
    segmenter = _ImageSegmenter.create_from_options(options)
    self.assertIsInstance(segmenter, _ImageSegmenter)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name' or 'file_descriptor_meta'."):
      base_options = _BaseOptions(file_name='')
      options = _ImageSegmenterOptions(base_options=base_options)
      _ImageSegmenter.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(file_content=f.read())
      options = _ImageSegmenterOptions(base_options=base_options)
      segmenter = _ImageSegmenter.create_from_options(options)
      self.assertIsInstance(segmenter, _ImageSegmenter)

  @parameterized.parameters(
      (ModelFileType.FILE_NAME, _EXPECTED_COLORED_LABELS),
      (ModelFileType.FILE_CONTENT, _EXPECTED_COLORED_LABELS))
  def test_segment_model(self, model_file_type, expected_colored_labels):
    # Creates segmenter.
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=self.model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(self.model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    segmenter = _create_segmenter_from_options(base_options)

    # Loads image.
    image = tensor_image.TensorImage.create_from_file(self.test_image_path)

    # Performs image segmentation on the input.
    segmentation = segmenter.segment(image).segmentation[0]
    colored_labels = segmentation.colored_labels

    # Check if the sizes of the result and expected colored labels are the same.
    self.assertEqual(
        len(colored_labels), len(expected_colored_labels),
        'Number of colored labels do not match.')

    # Comparing results.
    for index in range(len(expected_colored_labels)):
      for key in expected_colored_labels[index].keys():
        self.assertEqual(
            getattr(colored_labels[index], key),
            expected_colored_labels[index][key])

  def test_segmentation_category_mask(self):
    """Check if category mask matches with ground truth."""
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

    self.assertEqual(
        len(result_pixels), len(ground_truth_pixels),
        'Segmentation mask size does not match the ground truth mask size.')

    inconsistent_pixels = 0

    for index in range(num_pixels):
      inconsistent_pixels += (
          result_pixels[index] * _MASK_MAGNIFICATION_FACTOR !=
          ground_truth_pixels[index])

    self.assertLessEqual(
        inconsistent_pixels / num_pixels, _MATCH_PIXELS_THRESHOLD,
        f'Number of pixels in the candidate mask differing from that of the '
        f'ground truth mask exceeds {_MATCH_PIXELS_THRESHOLD}.')

  def test_segmentation_confidence_mask_matches_category_mask(self):
    """Check if the confidence mask matches with the category mask."""
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

    # Run segmentation on the model in CONFIDENCE_MASK mode.
    segmenter = _create_segmenter_from_options(
        base_options, output_type=_OutputType.CONFIDENCE_MASK)

    # Performs image segmentation on the input again.
    segmentation = segmenter.segment(image).segmentation[0]

    # Gets the list of confidence masks and colored_labels.
    confidence_masks = segmentation.confidence_masks.confidence_mask
    colored_labels = segmentation.colored_labels

    # Check if confidence mask shape is correct.
    self.assertEqual(
        len(confidence_masks), len(colored_labels),
        'Number of confidence masks must match with number of categories.')

    # Gather the confidence masks in a single array `confidence_mask_array`.
    confidence_mask_array = np.array([
        confidence_masks[index].value for index in range(len(confidence_masks))
    ])

    # Compute the category mask from the created confidence mask.
    calculated_category_mask = np.argmax(confidence_mask_array, axis=0)
    self.assertListEqual(
        calculated_category_mask.tolist(), category_mask.tolist(),
        'Confidence mask does not match with the category mask.')


if __name__ == '__main__':
  tf.test.main()
