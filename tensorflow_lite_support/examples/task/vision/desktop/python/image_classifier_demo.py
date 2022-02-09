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
# ==============================================================================
"""Python demo tool for Image Classification."""

import json

from absl import app
from absl import flags
from google.protobuf import json_format

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor.proto import classification_options_pb2
from tensorflow_lite_support.python.task.vision import image_classifier
from tensorflow_lite_support.python.task.vision.core import tensor_image

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', None,
                    'Absolute path to the ".tflite" image classifier model.')
flags.DEFINE_string(
    'image_path', None,
    'Absolute path to the image to classify. The image must be RGB or '
    'RGBA (grayscale is not supported). The image EXIF orientation '
    'flag, if any, is NOT taken into account.')
flags.DEFINE_integer('max_results', 5,
                     'Maximum number of classification results to display.')
flags.DEFINE_float(
    'score_threshold', 0,
    'Classification results with a confidence score below this value are '
    'rejected. If >= 0, overrides the score threshold(s) provided in the '
    'TFLite Model Metadata. Ignored otherwise.')
flags.DEFINE_string(
    'class_name_allowlist', '',
    'Comma-separated list of class names that acts as a whitelist. If '
    'non-empty, classification results whose "class_name" is not in this list '
    'are filtered out. Mutually exclusive with "class_name_denylist".')
flags.DEFINE_string(
    'class_name_denylist', '',
    'Comma-separated list of class names that acts as a blacklist. If '
    'non-empty, classification results whose "class_name" is in this list '
    'are filtered out. Mutually exclusive with "class_name_allowlist".')
flags.DEFINE_bool(
    'use_coral', False,
    'If true, inference will be delegated to a connected Coral Edge TPU '
    'device.')


def build_options():
  base_options = task_options.BaseOptions(
    model_file=task_options.ExternalFile(file_name=FLAGS.model_path),
    use_coral=FLAGS.use_coral)
  classification_options = classification_options_pb2.ClassificationOptions(
    max_results=FLAGS.max_results, score_threshold=FLAGS.score_threshold,
    class_name_allowlist=FLAGS.class_name_allowlist,
    class_name_denylist=FLAGS.class_name_denylist, use_coral=FLAGS.use_coral)
  return image_classifier.ImageClassifierOptions(
    base_options=base_options, classification_options=classification_options)


def main(_) -> None:
  # Creates classifier.
  options = build_options()
  classifier = image_classifier.ImageClassifier.create_from_options(options)

  # Loads image.
  image = tensor_image.TensorImage.from_file(FLAGS.image_path)

  # Run classification.
  result = classifier.classify(image)

  # Gets results.
  image_result_dict = json.loads(json_format.MessageToJson(result))
  print(image_result_dict)


if __name__ == "__main__":
  flags.mark_flag_as_required('model_path')
  flags.mark_flag_as_required('image_path')
  app.run(main)
