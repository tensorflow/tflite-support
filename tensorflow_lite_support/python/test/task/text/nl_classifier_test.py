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
"""Tests for nl_classifier."""

import enum

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_lite_support.python.task.core.proto import base_options_pb2
from tensorflow_lite_support.python.task.processor.proto import nl_classification_options_pb2
from tensorflow_lite_support.python.task.text import nl_classifier
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_pb2.BaseOptions
_NLClassifier = nl_classifier.NLClassifier
_NLClassifierOptions = nl_classifier.NLClassifierOptions
_NLClassificationOptions = nl_classification_options_pb2.NLClassificationOptions

_INPUT_STR = 'hello'

_TEST_MODEL = 'test_model_nl_classifier.tflite'

_DEFAULT_INPUT_TENSOR_NAME = _INPUT_TENSOR_NAME = 'INPUT'
_OUTPUT_DEQUANTIZED_TENSOR_NAME = 'OUTPUT_SCORE_DEQUANTIZED'
_OUTPUT_DEQUANTIZED_TENSOR_FLOAT64_NAME = 'OUTPUT_SCORE_DEQUANTIZED_FLOAT64'
_OUTPUT_QUANTIZED_TENSOR_NAME = 'OUTPUT_SCORE_QUANTIZED'
_OUTPUT_LABEL_TENSOR_NAME = 'LABELS'
_DEFAULT_OUTPUT_LABEL_TENSOR_NAME = 'OUTPUT_LABEL'

_OUTPUT_DEQUANTIZED_TENSOR_INDEX = 0
_OUTPUT_QUANTIZED_TENSOR_INDEX = 1
_OUTPUT_LABEL_TENSOR_INDEX = 2
_OUTPUT_DEQUANTIZED_TENSOR_FLOAT64_INDEX = 3
_DEFAULT_INPUT_TENSOR_INDEX = 0
_DEFAULT_OUTPUT_LABEL_TENSOR_INDEX = -1

_REGEX_TOKENIZER_MODEL = 'test_model_nl_classifier_with_regex_tokenizer.tflite'
_POSITIVE_INPUT = "This is the best movie I’ve seen in recent years. " \
                  "Strongly recommend it!"
_EXPECTED_RESULTS_OF_POSITIVE_INPUT = """
classifications {
  classes {
    score: 0.48657345771789551
    class_name: "Negative"
  }
  classes {
    score: 0.51342660188674927
    class_name: "Positive"
  }
  head_index: 0
}
"""
_NEGATIVE_INPUT = "What a waste of my time."
_EXPECTED_RESULTS_OF_NEGATIVE_INPUT = """
classifications {
  classes {
    score: 0.81312954425811768
    class_name: "Negative"
  }
  classes {
    score: 0.18687039613723755
    class_name: "Positive"
  }
  head_index: 0
}
"""

_LABEL_CUSTOM_OPS_MODEL = 'test_model_nl_classifier_with_associated_label.tflite'
_EXPECTED_RESULTS_OF_LABEL_CUSTOM_OPS_MODEL = """
classifications {
  classes {
    score: 255.0
    class_name: "label0"
  }
  classes {
    score: 510.0
    class_name: "label1"
  }
  classes {
    score: 765.0
    class_name: "label2"
  }
  head_index: 0
}
"""

_LABEL_BUILTIN_OPS_MODEL = 'test_model_nl_classifier_with_associated_label_builtin_ops.tflite'
_EXPECTED_RESULTS_OF_LABEL_BUILTIN_OPS_MODEL = """
classifications {
  classes {
    score: 0.49332118034362793
    class_name: "Negative"
  }
  classes {
    score: 0.50667881965637207
    class_name: "Positive"
  }
  head_index: 0
}
"""

_BOOLEAN_OUTPUT_MODEL = 'test_model_nl_classifier_bool_output.tflite'
_EXPECTED_RESULTS_OF_BOOLEAN_OUTPUT_MODEL = """
classifications {
  classes {
    score: 1.0
    class_name: "0"
  }
  classes {
    score: 1.0
    class_name: "1"
  }
  classes {
    score: 0.0
    class_name: "2"
  }
  head_index: 0
}
"""


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class NLClassifierTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.test_model_path = test_util.get_test_data_path(_TEST_MODEL)
    self.regex_model_path = test_util.get_test_data_path(_REGEX_TOKENIZER_MODEL)
    self.label_builtin_ops_path = \
      test_util.get_test_data_path(_LABEL_BUILTIN_OPS_MODEL)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    classifier = _NLClassifier.create_from_file(self.test_model_path)
    self.assertIsInstance(classifier, _NLClassifier)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(file_name=self.test_model_path)
    options = _NLClassifierOptions(base_options=base_options)
    classifier = _NLClassifier.create_from_options(options)
    self.assertIsInstance(classifier, _NLClassifier)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name' or 'file_descriptor_meta'."):
      base_options = _BaseOptions(file_name='')
      options = _NLClassifierOptions(base_options=base_options)
      _NLClassifier.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.test_model_path, 'rb') as f:
      base_options = _BaseOptions(file_content=f.read())
      options = _NLClassifierOptions(base_options=base_options)
      classifier = _NLClassifier.create_from_options(options)
      self.assertIsInstance(classifier, _NLClassifier)

  @parameterized.parameters(
      (_DEFAULT_INPUT_TENSOR_NAME, _OUTPUT_QUANTIZED_TENSOR_NAME,
       _OUTPUT_LABEL_TENSOR_NAME),
      (_DEFAULT_INPUT_TENSOR_NAME, _OUTPUT_QUANTIZED_TENSOR_NAME,
       _DEFAULT_OUTPUT_LABEL_TENSOR_NAME),
      (_DEFAULT_INPUT_TENSOR_NAME, _OUTPUT_DEQUANTIZED_TENSOR_NAME,
       _OUTPUT_LABEL_TENSOR_NAME),
      (_DEFAULT_INPUT_TENSOR_NAME, _OUTPUT_DEQUANTIZED_TENSOR_NAME,
       _DEFAULT_OUTPUT_LABEL_TENSOR_NAME),
      (_DEFAULT_INPUT_TENSOR_NAME, _OUTPUT_DEQUANTIZED_TENSOR_FLOAT64_NAME,
       _OUTPUT_LABEL_TENSOR_NAME),
      (_DEFAULT_INPUT_TENSOR_NAME, _OUTPUT_DEQUANTIZED_TENSOR_FLOAT64_NAME,
       _DEFAULT_OUTPUT_LABEL_TENSOR_NAME))
  def test_create_from_options_succeeds_with_tensor_name(self,
      input_tensor_name, output_score_tensor_name, output_label_tensor_name):
    # Test the API with different combinations of tensor name in creating proto
    # NLClassifierOptions
    base_options = _BaseOptions(file_name=self.test_model_path)
    nl_classification_options = _NLClassificationOptions(
      input_tensor_name=input_tensor_name,
      output_score_tensor_name=output_score_tensor_name,
      output_label_tensor_name=output_label_tensor_name)
    options = _NLClassifierOptions(
      base_options=base_options,
      nl_classification_options=nl_classification_options)
    _NLClassifier.create_from_options(options)

  @parameterized.parameters(
      (_DEFAULT_INPUT_TENSOR_INDEX, _OUTPUT_QUANTIZED_TENSOR_INDEX,
       _OUTPUT_LABEL_TENSOR_INDEX),
      (_DEFAULT_INPUT_TENSOR_INDEX, _OUTPUT_QUANTIZED_TENSOR_INDEX,
       _DEFAULT_OUTPUT_LABEL_TENSOR_INDEX),
      (_DEFAULT_INPUT_TENSOR_INDEX, _OUTPUT_DEQUANTIZED_TENSOR_INDEX,
       _OUTPUT_LABEL_TENSOR_INDEX),
      (_DEFAULT_INPUT_TENSOR_INDEX, _OUTPUT_DEQUANTIZED_TENSOR_INDEX,
       _DEFAULT_OUTPUT_LABEL_TENSOR_INDEX),
      (_DEFAULT_INPUT_TENSOR_INDEX, _OUTPUT_DEQUANTIZED_TENSOR_FLOAT64_INDEX,
       _OUTPUT_LABEL_TENSOR_INDEX),
      (_DEFAULT_INPUT_TENSOR_INDEX, _OUTPUT_DEQUANTIZED_TENSOR_FLOAT64_INDEX,
       _DEFAULT_OUTPUT_LABEL_TENSOR_INDEX))
  def test_create_from_options_succeeds_with_tensor_index(self,
      input_tensor_index, output_score_tensor_index, output_label_tensor_index):
    # Test the API with different combinations of tensor index in creating proto
    # NLClassifierOptions
    base_options = _BaseOptions(file_name=self.test_model_path)
    nl_classification_options = _NLClassificationOptions(
      input_tensor_index=input_tensor_index,
      output_score_tensor_index=output_score_tensor_index,
      output_label_tensor_index=output_label_tensor_index)
    options = _NLClassifierOptions(
      base_options=base_options,
      nl_classification_options=nl_classification_options)
    _NLClassifier.create_from_options(options)

  def test_create_from_options_fails_with_incorrect_input_tensor(self):
    # Invalid input tensor name.
    with self.assertRaisesRegex(
        ValueError,
        r"No input tensor found with name I do not exist or at index -1"):
      base_options = _BaseOptions(file_name=self.test_model_path)
      nl_classification_options = _NLClassificationOptions(
        input_tensor_index=-1, input_tensor_name='I do not exist')
      options = _NLClassifierOptions(
        base_options=base_options,
        nl_classification_options=nl_classification_options)
      _NLClassifier.create_from_options(options)

  def test_create_from_options_fails_with_incorrect_output_score_tensor(self):
    # Invalid output score tensor name.
    with self.assertRaisesRegex(
        ValueError,
        r"No output score tensor found with name "
        r"invalid_tensor or at index -1"):
      base_options = _BaseOptions(file_name=self.test_model_path)
      nl_classification_options = _NLClassificationOptions(
        output_score_tensor_index=-1, output_score_tensor_name='invalid_tensor')
      options = _NLClassifierOptions(
        base_options=base_options,
        nl_classification_options=nl_classification_options)
      _NLClassifier.create_from_options(options)

  @parameterized.parameters(
      # Regex tokenizer model.
      (_REGEX_TOKENIZER_MODEL, ModelFileType.FILE_NAME, _POSITIVE_INPUT,
       _EXPECTED_RESULTS_OF_POSITIVE_INPUT),
      (_REGEX_TOKENIZER_MODEL, ModelFileType.FILE_NAME, _NEGATIVE_INPUT,
       _EXPECTED_RESULTS_OF_NEGATIVE_INPUT),
      (_REGEX_TOKENIZER_MODEL, ModelFileType.FILE_CONTENT, _POSITIVE_INPUT,
       _EXPECTED_RESULTS_OF_POSITIVE_INPUT),
      (_REGEX_TOKENIZER_MODEL, ModelFileType.FILE_CONTENT, _NEGATIVE_INPUT,
       _EXPECTED_RESULTS_OF_NEGATIVE_INPUT),
      # Model with label custom ops.
      (_LABEL_CUSTOM_OPS_MODEL, ModelFileType.FILE_NAME, _INPUT_STR,
       _EXPECTED_RESULTS_OF_LABEL_CUSTOM_OPS_MODEL),
      (_LABEL_CUSTOM_OPS_MODEL, ModelFileType.FILE_CONTENT, _INPUT_STR,
       _EXPECTED_RESULTS_OF_LABEL_CUSTOM_OPS_MODEL),
      # Model with label builtin ops.
      (_LABEL_BUILTIN_OPS_MODEL, ModelFileType.FILE_NAME, _INPUT_STR,
       _EXPECTED_RESULTS_OF_LABEL_BUILTIN_OPS_MODEL),
      (_LABEL_BUILTIN_OPS_MODEL, ModelFileType.FILE_CONTENT, _INPUT_STR,
       _EXPECTED_RESULTS_OF_LABEL_BUILTIN_OPS_MODEL),
      # Model with boolean outputs.
      (_BOOLEAN_OUTPUT_MODEL, ModelFileType.FILE_NAME, _INPUT_STR,
       _EXPECTED_RESULTS_OF_BOOLEAN_OUTPUT_MODEL),
      (_BOOLEAN_OUTPUT_MODEL, ModelFileType.FILE_CONTENT, _INPUT_STR,
       _EXPECTED_RESULTS_OF_BOOLEAN_OUTPUT_MODEL))
  def test_classify_model(self, model_name, model_file_type, text,
                          expected_result_text_proto):
    # Creates classifier.
    model_path = test_util.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, "rb") as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _NLClassifierOptions(base_options=base_options)
    classifier = _NLClassifier.create_from_options(options)
    # Classifies text using the given model.
    text_classification_result = classifier.classify(text)
    self.assertProtoEquals(expected_result_text_proto,
                           text_classification_result)


if __name__ == "__main__":
  tf.test.main()
