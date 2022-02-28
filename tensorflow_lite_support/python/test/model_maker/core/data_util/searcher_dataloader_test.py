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
"""Tests for searcher_dataloader."""

import numpy as np
from tensorflow_lite_support.python.model_maker.core.data_util import searcher_dataloader
from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.vision import image_embedder
from tensorflow_lite_support.python.test import test_util
import unittest

_BaseOptions = task_options.BaseOptions
_ExternalFile = task_options.ExternalFile
_ImageEmbedder = image_embedder.ImageEmbedder
_ImageEmbedderOptions = image_embedder.ImageEmbedderOptions
_SearcherDataLoader = searcher_dataloader.DataLoader


class SearcherDataloaderTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    tflite_path = test_util.get_test_data_path(
        "mobilenet_v3_small_100_224_embedder.tflite")
    base_options = _BaseOptions(model_file=_ExternalFile(file_name=tflite_path))
    options = _ImageEmbedderOptions(base_options=base_options)
    self.embedder = _ImageEmbedder.create_from_options(options)

  def test_concat_dataset(self):
    dataset1 = np.random.rand(2, 1024)
    dataset2 = np.random.rand(2, 1024)
    dataset3 = np.random.rand(1, 1024)
    metadata = ["0", "1", b"\x11\x22", b"\x33\x44", "4"]
    data = _SearcherDataLoader(embedder=self.embedder)
    data._dataset = dataset1
    data._cache_dataset_list = [dataset2, dataset3]
    data._metadata = metadata

    self.assertTrue((data.dataset == np.vstack([dataset1, dataset2,
                                                dataset3])).all())
    self.assertEqual(data.metadata, metadata)

  def test_append(self):
    dataset1 = np.random.rand(4, 1024)
    metadata1 = ["0", "1", b"\x11\x22", "3"]
    data_loader1 = _SearcherDataLoader(embedder=self.embedder)
    data_loader1._dataset = dataset1
    data_loader1._metadata = metadata1

    dataset2 = np.random.rand(2, 1024)
    metadata2 = [b"\x33\x44", "5"]
    data_loader2 = _SearcherDataLoader(embedder=self.embedder)
    data_loader2._dataset = dataset2
    data_loader2._metadata = metadata2

    data_loader1.append(data_loader2)
    self.assertEqual(data_loader1.dataset.shape, (6, 1024))
    self.assertEqual(data_loader1.metadata,
                     ["0", "1", b"\x11\x22", "3", b"\x33\x44", "5"])


if __name__ == "__main__":
  unittest.main()
