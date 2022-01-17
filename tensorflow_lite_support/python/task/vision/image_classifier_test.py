# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for image_classifier."""

from absl.testing import parameterized

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor import processor_options
from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2
from tensorflow_lite_support.python.task.processor.proto import classifications_pb2
from tensorflow_lite_support.python.task.vision import image_classifier
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.test import test_util
import unittest
import textwrap

_MODEL_FLOAT = "mobilenet_v2_1.0_224.tflite"
_MODEL_QUANTIZED = "mobilenet_v1_0.25_224_quant.tflite"


class ImageClassifierTest(parameterized.TestCase, unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Float model path
    self.model_path = test_util.get_test_data_path(
      _MODEL_FLOAT)
    # Quantized model path
    self.quantized_model_path = test_util.get_test_data_path(
      _MODEL_QUANTIZED)

  def test_create_from_options(self):
    # Creates with options containing model file successfully.
    base_options = task_options.BaseOptions(model_file=self.model_path)
    options = image_classifier.ImageClassifierOptions(base_options=base_options)
    image_classifier.ImageClassifier.create_from_options(options)

    # Missing the model file.
    with self.assertRaisesRegex(
            TypeError,
            r"__init__\(\) missing 1 required positional argument: 'model_file'"):
      base_options = task_options.BaseOptions()

    # Invalid empty model path.
    with self.assertRaisesRegex(
            Exception,
            r"INVALID_ARGUMENT: ExternalFile must specify at least one of "
            r"'file_content', file_name' or 'file_descriptor_meta'\. "
            r"\[tflite::support::TfLiteSupportStatus='2'\]"):
      base_options = task_options.BaseOptions(model_file="")
      options = image_classifier.ImageClassifierOptions(base_options=base_options)
      image_classifier.ImageClassifier.create_from_options(options)

    # Invalid max results.
    with self.assertRaisesRegex(
            Exception,
            r"INVALID_ARGUMENT: Invalid `max_results` option: value must be != 0 "
            r"\[tflite::support::TfLiteSupportStatus='2'\]"):
      base_options = task_options.BaseOptions(model_file=self.model_path)
      classifier_options = processor_options.ClassificationOptions(
        max_results=0)
      options = image_classifier.ImageClassifierOptions(
        base_options=base_options,
        classifier_options=classifier_options)
      image_classifier.ImageClassifier.create_from_options(options)

  @parameterized.parameters(
    (['foo'], ['bar']),
  )
  def test_combined_whitelist_and_blacklist(self, class_name_whitelist,
                                            class_name_blacklist):
    # Fails with combined whitelist and blacklist
    with self.assertRaisesRegex(
            Exception,
            r"INVALID_ARGUMENT: `class_name_whitelist` and `class_name_blacklist` "
            r"are mutually exclusive options. "
            r"\[tflite::support::TfLiteSupportStatus='2'\]"):
      base_options = task_options.BaseOptions(model_file=self.model_path)
      classifier_options = processor_options.ClassificationOptions(
        class_name_whitelist=class_name_whitelist,
        class_name_blacklist=class_name_blacklist)
      options = image_classifier.ImageClassifierOptions(
        base_options=base_options,
        classifier_options=classifier_options)
      image_classifier.ImageClassifier.create_from_options(options)

  @parameterized.parameters(
    (3, None, None, None, False),
  )
  def test_classify_float_model(self, max_results, score_threshold,
                                class_name_whitelist, class_name_blacklist,
                                with_bounding_box):
    # Creates classifier.
    base_options = task_options.BaseOptions(model_file=self.model_path)
    classifier_options = processor_options.ClassificationOptions(
      max_results=max_results,
      score_threshold=score_threshold,
      class_name_whitelist=class_name_whitelist,
      class_name_blacklist=class_name_blacklist)
    options = image_classifier.ImageClassifierOptions(
      base_options=base_options, classifier_options=classifier_options)
    classifier = image_classifier.ImageClassifier(options)

    # Loads images: one is a crop of the other.
    image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger.jpg"))
    cropped_image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger_crop.jpg"))

    bounding_box = None
    if with_bounding_box:
      # Bounding box in "burger.jpg" corresponding to "burger_crop.jpg".
      bounding_box = bounding_box_pb2.BoundingBox(
        origin_x=0, origin_y=0, width=400, height=325)

    # Classifies both inputs.
    image_result = classifier.classify(image, bounding_box)
    crop_result = classifier.classify(cropped_image)

    self.assertMultiLineEqual(
      str(image_result),
      textwrap.dedent(
        """\
        classifications {
          classes {
            index: 934
            score: 0.7399742007255554
            class_name: "cheeseburger"
          }
          classes {
            index: 925
            score: 0.026928534731268883
            class_name: "guacamole"
          }
          classes {
            index: 932
            score: 0.025737214833498
            class_name: "bagel"
          }
          head_index: 0
        }
        """)
    )
    self.assertMultiLineEqual(
      str(crop_result),
      textwrap.dedent(
        """\
        classifications {
          classes {
            index: 934
            score: 0.8810749650001526
            class_name: "cheeseburger"
          }
          classes {
            index: 925
            score: 0.019916774705052376
            class_name: "guacamole"
          }
          classes {
            index: 932
            score: 0.012394513003528118
            class_name: "bagel"
          }
          head_index: 0
        }
        """)
    )


if __name__ == "__main__":
  unittest.main()
