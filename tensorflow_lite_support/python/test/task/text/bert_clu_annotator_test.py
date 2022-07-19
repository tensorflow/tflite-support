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
"""Tests for bert_clu_annotator."""

import enum

from absl.testing import parameterized

import tensorflow as tf
from tensorflow_lite_support.python.task.core import base_options as base_options_module
from tensorflow_lite_support.python.task.processor.proto import class_pb2
from tensorflow_lite_support.python.task.processor.proto import clu_pb2
from tensorflow_lite_support.python.task.processor.proto import clu_annotation_options_pb2
from tensorflow_lite_support.python.task.text import bert_clu_annotator
from tensorflow_lite_support.python.test import test_util

_BaseOptions = base_options_module.BaseOptions
_BertCluAnnotator = bert_clu_annotator.BertCluAnnotator
_CluRequest = clu_pb2.CluRequest
_CluResponse = clu_pb2.CluResponse
_CategoricalSlot = clu_pb2.CategoricalSlot
_Extraction = clu_pb2.Extraction
_NonCategoricalSlot = clu_pb2.NonCategoricalSlot
_Category = class_pb2.Category
_BertCluAnnotatorOptions = bert_clu_annotator.BertCluAnnotatorOptions
_BertCluAnnotationOptions = clu_annotation_options_pb2.BertCluAnnotationOptions

_BERT_MODEL = 'bert_clu_annotator_with_metadata.tflite'
_CLU_REQUEST = _CluRequest(utterances=[
    'I would like to make a restaurant reservation at morning 11:15.',
    'Which restaurant do you want to go to?',
    'Can I get a reservation for two people at Andes Cafe? Where is their ' +
    'address?'
])
_CLU_RESPONSE = _CluResponse(
    domains=[
        _Category(
            index=0,
            score=0.925057,
            display_name='Restaurants',
            category_name='')
    ],
    intents=[],
    categorical_slots=[
        _CategoricalSlot(
            slot='number_of_seats',
            prediction=_Category(
                index=0, score=0.920792, display_name='2', category_name=''))
    ],
    noncategorical_slots=[
        _NonCategoricalSlot(
            slot='restaurant_name',
            extraction=_Extraction(
                value='Andes Cafe', score=0.914949, start=42, end=52))
    ])


class ModelFileType(enum.Enum):
  FILE_CONTENT = 1
  FILE_NAME = 2


class BertCLUAnnotatorTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(_BERT_MODEL)

  def test_create_from_file_succeeds_with_valid_model_path(self):
    # Creates with default option and valid model file successfully.
    annotator = _BertCluAnnotator.create_from_file(self.model_path)
    self.assertIsInstance(annotator, _BertCluAnnotator)

  def test_create_from_options_succeeds_with_valid_model_path(self):
    # Creates with options containing model file successfully.
    base_options = _BaseOptions(file_name=self.model_path)
    options = _BertCluAnnotatorOptions(base_options=base_options)
    annotator = _BertCluAnnotator.create_from_options(options)
    self.assertIsInstance(annotator, _BertCluAnnotator)

  def test_create_from_options_fails_with_invalid_model_path(self):
    # Invalid empty model path.
    with self.assertRaisesRegex(
        ValueError,
        r"ExternalFile must specify at least one of 'file_content', "
        r"'file_name' or 'file_descriptor_meta'."):
      base_options = _BaseOptions(file_name='')
      options = _BertCluAnnotatorOptions(base_options=base_options)
      _BertCluAnnotator.create_from_options(options)

  def test_create_from_options_succeeds_with_valid_model_content(self):
    # Creates with options containing model content successfully.
    with open(self.model_path, 'rb') as f:
      base_options = _BaseOptions(file_content=f.read())
      options = _BertCluAnnotatorOptions(base_options=base_options)
      annotator = _BertCluAnnotator.create_from_options(options)
      self.assertIsInstance(annotator, _BertCluAnnotator)

  @parameterized.parameters(
      (_BERT_MODEL, ModelFileType.FILE_NAME, _CLU_REQUEST, _CLU_RESPONSE),
      (_BERT_MODEL, ModelFileType.FILE_CONTENT, _CLU_REQUEST, _CLU_RESPONSE),
  )
  def test_annotate_model(self, model_name, model_file_type, clu_request,
                          expected_clu_response):
    # Creates annotator.
    model_path = test_util.get_test_data_path(model_name)
    if model_file_type is ModelFileType.FILE_NAME:
      base_options = _BaseOptions(file_name=model_path)
    elif model_file_type is ModelFileType.FILE_CONTENT:
      with open(model_path, 'rb') as f:
        model_content = f.read()
      base_options = _BaseOptions(file_content=model_content)
    else:
      # Should never happen
      raise ValueError('model_file_type is invalid.')

    options = _BertCluAnnotatorOptions(base_options=base_options)
    annotator = _BertCluAnnotator.create_from_options(options)

    # Annotates CLU request using the given model.
    text_clu_response = annotator.annotate(clu_request)
    self.assertProtoEquals(text_clu_response.to_pb2(),
                           expected_clu_response.to_pb2())

  @parameterized.parameters(
    (_CLU_REQUEST, _CluResponse(
      domains=[], intents=[], categorical_slots=[], noncategorical_slots=[
      _NonCategoricalSlot(
        slot='restaurant_name',
        extraction=_Extraction(
          value='Andes Cafe', score=0.914949, start=42, end=52))]),
     0.99, None, 0.99, None),
    (_CLU_REQUEST, _CluResponse(
        domains=[], intents=[], categorical_slots=[], noncategorical_slots=[]),
     0.99, None, 0.99, 0.99),
  )
  def test_thresholds(self, clu_request, expected_clu_response,
                      domain_threshold, intent_threshold,
                      categorical_slot_threshold, noncategorical_slot_threshold):
    # Creates annotator.
    base_options = _BaseOptions(file_name=self.model_path)
    bert_clu_annotation_options = _BertCluAnnotationOptions(
        domain_threshold=domain_threshold, intent_threshold=intent_threshold,
        categorical_slot_threshold=categorical_slot_threshold,
        noncategorical_slot_threshold=noncategorical_slot_threshold)
    options = _BertCluAnnotatorOptions(
        base_options=base_options,
        bert_clu_annotation_options=bert_clu_annotation_options)
    annotator = _BertCluAnnotator.create_from_options(options)

    # Annotates CLU request using the given model.
    text_clu_response = annotator.annotate(clu_request)
    self.assertProtoEquals(text_clu_response.to_pb2(),
                           expected_clu_response.to_pb2())

  def test_max_history_turns(self):
    # Creates annotator.
    base_options = _BaseOptions(file_name=self.model_path)
    bert_clu_annotation_options = _BertCluAnnotationOptions(
        max_history_turns=10)
    options = _BertCluAnnotatorOptions(
        base_options=base_options,
        bert_clu_annotation_options=bert_clu_annotation_options)
    annotator = _BertCluAnnotator.create_from_options(options)

    # Annotates CLU request using the given model.
    expected_clu_response = _CluResponse(
        domains=[
            _Category(
                index=0, score=0.925057, display_name='Restaurants',
                category_name='')
        ],
        intents=[],
        categorical_slots=[
            _CategoricalSlot(
                slot='number_of_seats',
                prediction=_Category(
                    index=0, score=0.920792, display_name='2',
                    category_name=''))
        ],
        noncategorical_slots=[
            _NonCategoricalSlot(
                slot='restaurant_name',
                extraction=_Extraction(
                    value='Andes Cafe', score=0.914949,
                    start=42, end=52))
        ])
    text_clu_response = annotator.annotate(_CLU_REQUEST)
    self.assertProtoEquals(text_clu_response.to_pb2(),
                           expected_clu_response.to_pb2())


if __name__ == '__main__':
  tf.test.main()
