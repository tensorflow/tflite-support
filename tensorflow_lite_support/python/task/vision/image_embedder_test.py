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
"""Tests for image_embedder."""

from absl.testing import parameterized

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor import processor_options
from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2
from tensorflow_lite_support.python.task.processor.proto import embeddings_pb2
from tensorflow_lite_support.python.task.vision import image_embedder
from tensorflow_lite_support.python.task.vision.core import tensor_image
from tensorflow_lite_support.python.test import test_util
import unittest


class ImageEmbedderTest(parameterized.TestCase, unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.model_path = test_util.get_test_data_path(
        "mobilenet_v3_small_100_224_embedder.tflite")

  def test_create_from_options(self):
    # Creates with options containing model file successfully.
    base_options = task_options.BaseOptions(model_file=self.model_path)
    options = image_embedder.ImageEmbedderOptions(base_options=base_options)
    image_embedder.ImageEmbedder.create_from_options(options)

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
      options = image_embedder.ImageEmbedderOptions(base_options=base_options)
      image_embedder.ImageEmbedder.create_from_options(options)

  @parameterized.parameters(
      (None, None, False, 0.932738),
      (True, None, False, 0.932738),
      (True, True, False, 0.929717),
      (None, None, True, 0.999914),
  )
  def test_embed(self, l2_normalize, quantize, with_bounding_box,
                 expected_similarity):
    # Creates embedder.
    base_options = task_options.BaseOptions(model_file=self.model_path)
    embedding_options = processor_options.EmbeddingOptions(
        l2_normalize=l2_normalize, quantize=quantize)
    options = image_embedder.ImageEmbedderOptions(
        base_options=base_options, embedding_options=embedding_options)
    embedder = image_embedder.ImageEmbedder(options)

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

    # Extracts both embeddings.
    image_result = embedder.embed(image, bounding_box)
    crop_result = embedder.embed(cropped_image)

    # Checks results sizes.
    self.assertLen(image_result.embeddings, 1)
    image_feature_vector = image_result.embeddings[0].feature_vector
    self.assertLen(crop_result.embeddings, 1)
    crop_feature_vector = crop_result.embeddings[0].feature_vector
    if quantize:
      self.assertLen(image_feature_vector.value_string, 1024)
      self.assertLen(crop_feature_vector.value_string, 1024)
    else:
      self.assertLen(image_feature_vector.value_float, 1024)
      self.assertLen(crop_feature_vector.value_float, 1024)

    # Checks cosine similarity.
    similarity = embedder.cosine_similarity(image_feature_vector,
                                            crop_feature_vector)
    self.assertAlmostEqual(similarity, expected_similarity, places=6)

  def test_get_embedding_by_index(self):
    base_options = task_options.BaseOptions(model_file=self.model_path)
    options = image_embedder.ImageEmbedderOptions(base_options=base_options)
    embedder = image_embedder.ImageEmbedder.create_from_options(options)

    # Builds test data.
    embedding = embeddings_pb2.Embedding(output_index=0)
    embedding.feature_vector.value_float.append(1.0)
    embedding.feature_vector.value_float.append(0.0)
    embedding_result = embeddings_pb2.EmbeddingResult()
    embedding_result.embeddings.append(embedding)

    result0 = embedder.get_embedding_by_index(embedding_result, 0)
    self.assertEqual(result0.output_index, 0)
    self.assertEqual(result0.feature_vector.value_float[0], 1.0)
    self.assertEqual(result0.feature_vector.value_float[1], 0.0)

    with self.assertRaisesRegex(ValueError, r"Output index is out of bound\."):
      embedder.get_embedding_by_index(embedding_result, 1)

  def test_get_embedding_dimension(self):
    base_options = task_options.BaseOptions(model_file=self.model_path)
    options = image_embedder.ImageEmbedderOptions(base_options=base_options)
    embedder = image_embedder.ImageEmbedder.create_from_options(options)
    self.assertEqual(embedder.get_embedding_dimension(0), 1024)
    self.assertEqual(embedder.get_embedding_dimension(1), -1)

  def test_number_of_output_layers(self):
    base_options = task_options.BaseOptions(model_file=self.model_path)
    options = image_embedder.ImageEmbedderOptions(base_options=base_options)
    embedder = image_embedder.ImageEmbedder.create_from_options(options)
    self.assertEqual(embedder.number_of_output_layers, 1)


if __name__ == "__main__":
  unittest.main()
