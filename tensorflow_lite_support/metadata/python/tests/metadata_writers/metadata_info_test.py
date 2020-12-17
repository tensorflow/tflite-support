# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for metadata info classes."""

from absl.testing import parameterized

import tensorflow as tf

import flatbuffers
from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata import schema_py_generated as _schema_fb
from tensorflow_lite_support.metadata.python import metadata as _metadata
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_info
from tensorflow_lite_support.metadata.python.tests.metadata_writers import test_utils


class GeneralMdTest(tf.test.TestCase):

  _EXPECTED_GENERAL_META_JSON = "../testdata/general_meta.json"

  def test_create_metadata_should_succeed(self):
    general_md = metadata_info.GeneralMd(
        name="model",
        version="v1",
        description="A ML model.",
        author="TensorFlow",
        licenses="Apache")
    general_metadata = general_md.create_metadata()

    # Create the Flatbuffers object and convert it to the json format.
    builder = flatbuffers.Builder(0)
    builder.Finish(
        general_metadata.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_json = _metadata.convert_to_json(bytes(builder.Output()))

    expected_json = test_utils.load_file(self._EXPECTED_GENERAL_META_JSON, "r")
    self.assertEqual(metadata_json, expected_json)


class AssociatedFileMdTest(tf.test.TestCase):

  _EXPECTED_META_JSON = "../testdata/associated_file_meta.json"

  def test_create_metadata_should_succeed(self):
    file_md = metadata_info.AssociatedFileMd(
        file_path="label.txt",
        description="The label file.",
        file_type=_metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS,
        locale="en")
    file_metadata = file_md.create_metadata()

    # Create the Flatbuffers object and convert it to the json format.
    model_metadata = _metadata_fb.ModelMetadataT()
    model_metadata.associatedFiles = [file_metadata]
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_metadata.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_json = _metadata.convert_to_json(bytes(builder.Output()))

    expected_json = test_utils.load_file(self._EXPECTED_META_JSON, "r")
    self.assertEqual(metadata_json, expected_json)


class TensorMdTest(tf.test.TestCase, parameterized.TestCase):

  _TENSOR_NAME = "input"
  _TENSOR_DESCRIPTION = "The input tensor."
  _TENSOR_MIN = 0
  _TENSOR_MAX = 1
  _LABEL_FILE_EN = "labels.txt"
  _LABEL_FILE_CN = "labels_cn.txt"  # Locale label file in Chinese.
  _EXPECTED_FEATURE_TENSOR_JSON = "../testdata/feature_tensor_meta.json"
  _EXPECTED_IMAGE_TENSOR_JSON = "../testdata/image_tensor_meta.json"
  _EXPECTED_BOUNDING_BOX_TENSOR_JSON = "../testdata/bounding_box_tensor_meta.json"

  @parameterized.named_parameters(
      {
          "testcase_name": "feature_tensor",
          "content_type": _metadata_fb.ContentProperties.FeatureProperties,
          "golden_json": _EXPECTED_FEATURE_TENSOR_JSON
      }, {
          "testcase_name": "image_tensor",
          "content_type": _metadata_fb.ContentProperties.ImageProperties,
          "golden_json": _EXPECTED_IMAGE_TENSOR_JSON
      }, {
          "testcase_name": "bounding_box_tensor",
          "content_type": _metadata_fb.ContentProperties.BoundingBoxProperties,
          "golden_json": _EXPECTED_BOUNDING_BOX_TENSOR_JSON
      })
  def test_create_metadata_should_succeed(self, content_type, golden_json):
    associated_file1 = metadata_info.AssociatedFileMd(
        file_path=self._LABEL_FILE_EN, locale="en")
    associated_file2 = metadata_info.AssociatedFileMd(
        file_path=self._LABEL_FILE_CN, locale="cn")

    tensor_md = metadata_info.TensorMd(
        name=self._TENSOR_NAME,
        description=self._TENSOR_DESCRIPTION,
        min_values=[self._TENSOR_MIN],
        max_values=[self._TENSOR_MAX],
        content_type=content_type,
        associated_files=[associated_file1, associated_file2])
    tensor_metadata = tensor_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata(tensor_metadata))
    expected_json = test_utils.load_file(golden_json, "r")
    self.assertEqual(metadata_json, expected_json)


class InputImageTensorMdTest(tf.test.TestCase, parameterized.TestCase):

  _NAME = "image"
  _DESCRIPTION = "The input image."
  _NORM_MEAN = (0, 127.5, 255)
  _NORM_STD = (127.5, 127.5, 127.5)
  _COLOR_SPACE_TYPE = _metadata_fb.ColorSpaceType.RGB
  _EXPECTED_FLOAT_TENSOR_JSON = "../testdata/input_image_tensor_float_meta.json"
  _EXPECTED_UINT8_TENSOR_JSON = "../testdata/input_image_tensor_uint8_meta.json"
  _EXPECTED_UNSUPPORTED_TENSOR_JSON = "../testdata/input_image_tensor_unsupported_meta.json"

  @parameterized.named_parameters(
      {
          "testcase_name": "float",
          "tensor_type": _schema_fb.TensorType.FLOAT32,
          "golden_json": _EXPECTED_FLOAT_TENSOR_JSON
      }, {
          "testcase_name": "uint8",
          "tensor_type": _schema_fb.TensorType.UINT8,
          "golden_json": _EXPECTED_UINT8_TENSOR_JSON
      }, {
          "testcase_name": "unsupported_tensor_type",
          "tensor_type": _schema_fb.TensorType.INT16,
          "golden_json": _EXPECTED_UNSUPPORTED_TENSOR_JSON
      })
  def test_create_metadata_should_succeed(self, tensor_type, golden_json):
    tesnor_md = metadata_info.InputImageTensorMd(
        name=self._NAME,
        description=self._DESCRIPTION,
        norm_mean=list(self._NORM_MEAN),
        norm_std=list(self._NORM_STD),
        color_space_type=self._COLOR_SPACE_TYPE,
        tensor_type=tensor_type)
    tensor_metadata = tesnor_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata(tensor_metadata))
    expected_json = test_utils.load_file(golden_json, "r")
    self.assertEqual(metadata_json, expected_json)

  def test_init_should_throw_exception_with_incompatible_mean_and_std(self):
    norm_mean = [0]
    norm_std = [1, 2]
    with self.assertRaises(ValueError) as error:
      metadata_info.InputImageTensorMd(norm_mean=norm_mean, norm_std=norm_std)
    # TODO(b/175843689): Python version cannot be specified in Kokoro bazel test
    self.assertEqual(
        "norm_mean and norm_std are expected to be the same dim. But got " +
        "{} and {}".format(len(norm_mean), len(norm_std)), str(error.exception))


class ClassificationTensorMdTest(tf.test.TestCase, parameterized.TestCase):

  _NAME = "probability"
  _DESCRIPTION = "The classification result tensor."
  _LABEL_FILE_EN = "labels.txt"
  _LABEL_FILE_CN = "labels_cn.txt"  # Locale label file in Chinese.
  _EXPECTED_FLOAT_TENSOR_JSON = "../testdata/classification_tensor_float_meta.json"
  _EXPECTED_UINT8_TENSOR_JSON = "../testdata/classification_tensor_uint8_meta.json"
  _EXPECTED_UNSUPPORTED_TENSOR_JSON = "../testdata/classification_tensor_unsupported_meta.json"

  @parameterized.named_parameters(
      {
          "testcase_name": "float",
          "tensor_type": _schema_fb.TensorType.FLOAT32,
          "golden_json": _EXPECTED_FLOAT_TENSOR_JSON
      }, {
          "testcase_name": "uint8",
          "tensor_type": _schema_fb.TensorType.UINT8,
          "golden_json": _EXPECTED_UINT8_TENSOR_JSON
      }, {
          "testcase_name": "unsupported_tensor_type",
          "tensor_type": _schema_fb.TensorType.INT16,
          "golden_json": _EXPECTED_UNSUPPORTED_TENSOR_JSON
      })
  def test_create_metadata_should_succeed(self, tensor_type, golden_json):
    label_file_en = metadata_info.LabelFileMd(
        file_path=self._LABEL_FILE_EN, locale="en")
    label_file_cn = metadata_info.LabelFileMd(
        file_path=self._LABEL_FILE_CN, locale="cn")
    tesnor_md = metadata_info.ClassificationTensorMd(
        name=self._NAME,
        description=self._DESCRIPTION,
        label_files=[label_file_en, label_file_cn],
        tensor_type=tensor_type)
    tensor_metadata = tesnor_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata(tensor_metadata))
    expected_json = test_utils.load_file(golden_json, "r")
    self.assertEqual(metadata_json, expected_json)


class CategoryTensorMdTest(tf.test.TestCase, parameterized.TestCase):

  _NAME = "category"
  _DESCRIPTION = "The category tensor."
  _LABEL_FILE_EN = "labels.txt"
  _LABEL_FILE_CN = "labels_cn.txt"  # Locale label file in Chinese.
  _EXPECTED_TENSOR_JSON = "../testdata/category_tensor_float_meta.json"

  def test_create_metadata_should_succeed(self):
    label_file_en = metadata_info.LabelFileMd(
        file_path=self._LABEL_FILE_EN, locale="en")
    label_file_cn = metadata_info.LabelFileMd(
        file_path=self._LABEL_FILE_CN, locale="cn")
    tesnor_md = metadata_info.CategoryTensorMd(
        name=self._NAME,
        description=self._DESCRIPTION,
        label_files=[label_file_en, label_file_cn])
    tensor_metadata = tesnor_md.create_metadata()

    metadata_json = _metadata.convert_to_json(
        _create_dummy_model_metadata(tensor_metadata))
    expected_json = test_utils.load_file(self._EXPECTED_TENSOR_JSON, "r")
    self.assertEqual(metadata_json, expected_json)


def _create_dummy_model_metadata(
    tensor_metadata: _metadata_fb.TensorMetadataT) -> bytes:
  # Create a dummy model using the tensor metadata.
  subgraph_metadata = _metadata_fb.SubGraphMetadataT()
  subgraph_metadata.inputTensorMetadata = [tensor_metadata]
  model_metadata = _metadata_fb.ModelMetadataT()
  model_metadata.subgraphMetadata = [subgraph_metadata]

  # Create the Flatbuffers object and convert it to the json format.
  builder = flatbuffers.Builder(0)
  builder.Finish(
      model_metadata.Pack(builder),
      _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
  return bytes(builder.Output())


if __name__ == "__main__":
  tf.test.main()
