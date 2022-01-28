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
"""Tests for image_classifier."""

from absl.testing import parameterized

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor import processor_options
from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2
from tensorflow_lite_support.python.task.vision import image_classifier
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.test import test_util

import unittest
import textwrap


_MODEL_FLOAT = "mobilenet_v2_1.0_224.tflite"
_MODEL_QUANTIZED = "mobilenet_v1_0.25_224_quant.tflite"
_MODEL_AUTOML = "automl_labeler_model.tflite"


class ImageClassifierTest(parameterized.TestCase, unittest.TestCase):

  def setUp(self):
    super().setUp()

  @staticmethod
  def create_classifier_from_options(model_file, **classification_options):
    base_options = task_options.BaseOptions(model_file=model_file)
    classifier_options = processor_options.ClassificationOptions(
      **classification_options)
    options = image_classifier.ImageClassifierOptions(
      base_options=base_options, classifier_options=classifier_options)
    classifier = image_classifier.ImageClassifier(options)
    return classifier

  @parameterized.parameters(
    (_MODEL_FLOAT,),
    (_MODEL_QUANTIZED,),
    (_MODEL_AUTOML,),
  )
  def test_create_from_options(self, model_name):
    # Get the model path from the test data directory
    model_file = test_util.get_test_data_path(model_name)

    # Creates with options containing model file successfully.
    base_options = task_options.BaseOptions(model_file=model_file)
    options = image_classifier.ImageClassifierOptions(base_options=base_options)
    image_classifier.ImageClassifier.create_from_options(options)

    # Creates the classifier with the `num_threads` option successfully
    base_options = task_options.BaseOptions(model_file=model_file, num_threads=4)
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
      base_options = task_options.BaseOptions(model_file=model_file)
      classifier_options = processor_options.ClassificationOptions(
        max_results=0)
      options = image_classifier.ImageClassifierOptions(
        base_options=base_options,
        classifier_options=classifier_options)
      image_classifier.ImageClassifier.create_from_options(options)

    # Invalid number of threads.
    with self.assertRaisesRegex(
            Exception,
            r"INVALID_ARGUMENT: `num_threads` must be greater than 0 or equal to -1. "
            r"\[tflite::support::TfLiteSupportStatus='2']"):
      base_options = task_options.BaseOptions(model_file=model_file, num_threads=-2)
      options = image_classifier.ImageClassifierOptions(base_options=base_options)
      image_classifier.ImageClassifier.create_from_options(options)

  @parameterized.parameters(
    (_MODEL_FLOAT, ['foo'], ['bar']),
    (_MODEL_QUANTIZED, ['foo'], ['bar']),
    (_MODEL_AUTOML, ['foo'], ['bar']),
  )
  def test_combined_whitelist_and_blacklist(self, model_name,
                                            label_allowlist,
                                            label_denylist):
    # Get the model path from the test data directory
    model_file = test_util.get_test_data_path(model_name)

    # Fails with combined whitelist and blacklist
    with self.assertRaisesRegex(
            Exception,
            r"INVALID_ARGUMENT: `class_name_whitelist` and `class_name_blacklist` "
            r"are mutually exclusive options. "
            r"\[tflite::support::TfLiteSupportStatus='2'\]"):
      base_options = task_options.BaseOptions(model_file=model_file)
      classifier_options = processor_options.ClassificationOptions(
        label_allowlist=label_allowlist,
        label_denylist=label_denylist)
      options = image_classifier.ImageClassifierOptions(
        base_options=base_options,
        classifier_options=classifier_options)
      image_classifier.ImageClassifier.create_from_options(options)

  @parameterized.parameters(
    (_MODEL_FLOAT, 3, False),
    (_MODEL_QUANTIZED, 3, False),
    (_MODEL_AUTOML, 3, False),
    (_MODEL_FLOAT, 3, True),
    (_MODEL_QUANTIZED, 3, True),
    (_MODEL_AUTOML, 3, True),
  )
  def test_classify_model(self, model_name, max_results, with_bounding_box):
    # Get the model path from the test data directory
    model_file = test_util.get_test_data_path(model_name)

    # Creates classifier.
    classifier = self.create_classifier_from_options(
      model_file,
      max_results=max_results
    )

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

    if model_name == _MODEL_FLOAT:
      if with_bounding_box:
        # Testing the model on burger.jpg (w/ bounding box)
        self.assertEqual(
          str(image_result),
          textwrap.dedent(
            """\
            classifications {
              classes {
                index: 934
                score: 0.8815076351165771
                class_name: "cheeseburger"
              }
              classes {
                index: 925
                score: 0.019456762820482254
                class_name: "guacamole"
              }
              classes {
                index: 932
                score: 0.012489477172493935
                class_name: "bagel"
              }
              head_index: 0
            }
            """)
        )
      else:
        # Testing the model on burger.jpg (w/o bounding box)
        self.assertEqual(
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
      # Testing the model on burger_crop.jpg
      self.assertEqual(
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
    elif model_name == _MODEL_QUANTIZED:
      if with_bounding_box:
        # Testing the model on burger.jpg (w/ bounding box)
        self.assertEqual(
          str(image_result),
          textwrap.dedent(
            """\
            classifications {
              classes {
                index: 934
                score: 0.96484375
                class_name: "cheeseburger"
              }
              classes {
                index: 935
                score: 0.0078125
                class_name: "hotdog"
              }
              classes {
                index: 119
                score: 0.0078125
                class_name: "Dungeness crab"
              }
              head_index: 0
            }
            """)
        )
      else:
        # Testing the model on burger.jpg (w/o bounding box)
        self.assertEqual(
          str(image_result),
          textwrap.dedent(
            """\
            classifications {
              classes {
                index: 934
                score: 0.96484375
                class_name: "cheeseburger"
              }
              classes {
                index: 948
                score: 0.0078125
                class_name: "mushroom"
              }
              classes {
                index: 924
                score: 0.00390625
                class_name: "plate"
              }
              head_index: 0
            }
            """)
        )
      # Testing the model on burger_crop.jpg
      self.assertEqual(
        str(crop_result),
        textwrap.dedent(
          """\
          classifications {
            classes {
              index: 934
              score: 0.96484375
              class_name: "cheeseburger"
            }
            classes {
              index: 935
              score: 0.0078125
              class_name: "hotdog"
            }
            classes {
              index: 119
              score: 0.0078125
              class_name: "Dungeness crab"
            }
            head_index: 0
          }
          """)
      )
    elif model_name == _MODEL_AUTOML:
      if with_bounding_box:
        # Testing the model on burger.jpg (w/ bounding box)
        self.assertEqual(
          str(image_result),
          textwrap.dedent(
            """\
            classifications {
              classes {
                index: 2
                score: 0.953125
                class_name: "roses"
              }
              classes {
                index: 0
                score: 0.01171875
                class_name: "daisy"
              }
              classes {
                index: 1
                score: 0.01171875
                class_name: "dandelion"
              }
              head_index: 0
            }
            """)
        )
      else:
        # Testing the model on burger.jpg (w/o bounding box)
        self.assertEqual(
          str(image_result),
          textwrap.dedent(
            """\
            classifications {
              classes {
                index: 2
                score: 0.96484375
                class_name: "roses"
              }
              classes {
                index: 4
                score: 0.01171875
                class_name: "tulips"
              }
              classes {
                index: 0
                score: 0.0078125
                class_name: "daisy"
              }
              head_index: 0
            }
            """)
        )
      # Testing the model on burger_crop.jpg
      self.assertEqual(
        str(crop_result),
        textwrap.dedent(
          """\
          classifications {
            classes {
              index: 2
              score: 0.953125
              class_name: "roses"
            }
            classes {
              index: 0
              score: 0.01171875
              class_name: "daisy"
            }
            classes {
              index: 1
              score: 0.01171875
              class_name: "dandelion"
            }
            head_index: 0
          }
          """)
      )


  @parameterized.parameters(
    (_MODEL_FLOAT, None, 0.5, False),
  )
  def test_score_threshold_option(self, model_name,  max_results,
                                  score_threshold, with_bounding_box):
    # Get the model path from the test data directory
    model_file = test_util.get_test_data_path(model_name)

    # Creates classifier.
    classifier = self.create_classifier_from_options(
      model_file,
      max_results=max_results,
      score_threshold=score_threshold
    )

    # Loads image
    image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger.jpg"))

    bounding_box = None
    if with_bounding_box:
      # Bounding box in "burger.jpg" corresponding to "burger_crop.jpg".
      bounding_box = bounding_box_pb2.BoundingBox(
        origin_x=0, origin_y=0, width=400, height=325)

    # Classifies the input.
    image_result = classifier.classify(image, bounding_box)

    if model_name == _MODEL_FLOAT:
      # Testing the model on burger.jpg (w/o bounding box)
      self.assertEqual(
        str(image_result),
        textwrap.dedent(
          """\
          classifications {
            classes {
              index: 934
              score: 0.7399742007255554
              class_name: "cheeseburger"
            }
            head_index: 0
          }
          """)
      )

  @parameterized.parameters(
    (_MODEL_FLOAT, ['cheeseburger', 'guacamole']),
  )
  def test_whitelist_option(self, model_name, label_allowlist):
    # Get the model path from the test data directory
    model_file = test_util.get_test_data_path(model_name)

    # Creates classifier.
    classifier = self.create_classifier_from_options(
      model_file,
      label_allowlist=label_allowlist
    )

    # Loads image
    image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger.jpg"))

    # Classifies the input.
    image_result = classifier.classify(image, bounding_box=None)

    if model_name == _MODEL_FLOAT:
      # Testing the model on burger.jpg (w/o bounding box)
      self.assertEqual(
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
            head_index: 0
          }
          """)
      )

  @parameterized.parameters(
    (_MODEL_FLOAT, 0.01, ['cheeseburger', 'guacamole']),
  )
  def test_blacklist_option(self, model_name, score_threshold,
                            label_denylist):
    # Get the model path from the test data directory
    model_file = test_util.get_test_data_path(model_name)

    # Creates classifier.
    classifier = self.create_classifier_from_options(
      model_file,
      score_threshold=score_threshold,
      label_denylist=label_denylist
    )

    # Loads image
    image = tensor_image.TensorImage.from_file(
      test_util.get_test_data_path("burger.jpg"))

    # Classifies the input.
    image_result = classifier.classify(image, bounding_box=None)

    if model_name == _MODEL_FLOAT:
      # Testing the model on burger.jpg (w/o bounding box)
      self.assertEqual(
        str(image_result),
        textwrap.dedent(
          """\
          classifications {
            classes {
              index: 932
              score: 0.025737214833498
              class_name: "bagel"
            }
            classes {
              index: 963
              score: 0.010005592368543148
              class_name: "meat loaf"
            }
            head_index: 0
          }
          """)
      )


if __name__ == "__main__":
  unittest.main()
