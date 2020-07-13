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
"""Tests for tensorflow_lite_support.metadata.metadata."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import six

from flatbuffers.python import flatbuffers
from third_party.tensorflow.python.framework import test_util
from third_party.tensorflow.python.platform import resource_loader
from third_party.tensorflow.python.platform import test
from tensorflow_lite_support.metadata import metadata as _metadata
from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata import schema_py_generated as _schema_fb


class MetadataTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(MetadataTest, self).setUp()
    self._invalid_model_buf = None
    self._invalid_file = "not_existed_file"
    self._model_buf = self._create_model_buf()
    self._model_file = self.create_tempfile().full_path
    with open(self._model_file, "wb") as f:
      f.write(self._model_buf)
    self._metadata_file = self._create_metadata_file()
    self._metadata_file_with_version = self._create_metadata_file_with_version(
        self._metadata_file, "1.0.0")
    self._file1 = self.create_tempfile("file1").full_path
    self._file2 = self.create_tempfile("file2").full_path
    self._file3 = self.create_tempfile("file3").full_path

  def _create_model_buf(self):
    # Create a model with two inputs and one output, which matches the metadata
    # created by _create_metadata_file().
    metadata_field = _schema_fb.MetadataT()
    subgraph = _schema_fb.SubGraphT()
    subgraph.inputs = [0, 1]
    subgraph.outputs = [2]

    metadata_field.name = "meta"
    buffer_field = _schema_fb.BufferT()
    model = _schema_fb.ModelT()
    model.subgraphs = [subgraph]
    # Creates the metadata and buffer fields for testing purposes.
    model.metadata = [metadata_field, metadata_field]
    model.buffers = [buffer_field, buffer_field, buffer_field]
    model_builder = flatbuffers.Builder(0)
    model_builder.Finish(
        model.Pack(model_builder),
        _metadata.MetadataPopulator.TFLITE_FILE_IDENTIFIER)
    return model_builder.Output()

  def _create_metadata_file(self):
    associated_file1 = _metadata_fb.AssociatedFileT()
    associated_file1.name = b"file1"
    associated_file2 = _metadata_fb.AssociatedFileT()
    associated_file2.name = b"file2"
    self.expected_recorded_files = [
        six.ensure_str(associated_file1.name),
        six.ensure_str(associated_file2.name)
    ]

    input_meta = _metadata_fb.TensorMetadataT()
    output_meta = _metadata_fb.TensorMetadataT()
    output_meta.associatedFiles = [associated_file2]
    subgraph = _metadata_fb.SubGraphMetadataT()
    # Create a model with two inputs and one output.
    subgraph.inputTensorMetadata = [input_meta, input_meta]
    subgraph.outputTensorMetadata = [output_meta]

    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "Mobilenet_quantized"
    model_meta.associatedFiles = [associated_file1]
    model_meta.subgraphMetadata = [subgraph]
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)

    metadata_file = self.create_tempfile().full_path
    with open(metadata_file, "wb") as f:
      f.write(b.Output())
    return metadata_file

  def _create_model_buffer_with_wrong_identifier(self):
    wrong_identifier = b"widn"
    model = _schema_fb.ModelT()
    model_builder = flatbuffers.Builder(0)
    model_builder.Finish(model.Pack(model_builder), wrong_identifier)
    return model_builder.Output()

  def _create_metadata_buffer_with_wrong_identifier(self):
    # Creates a metadata with wrong identifier
    wrong_identifier = b"widn"
    metadata = _metadata_fb.ModelMetadataT()
    metadata_builder = flatbuffers.Builder(0)
    metadata_builder.Finish(metadata.Pack(metadata_builder), wrong_identifier)
    return metadata_builder.Output()

  def _populate_metadata_with_identifier(self, model_buf, metadata_buf,
                                         identifier):
    # For testing purposes only. MetadataPopulator cannot populate metadata with
    # wrong identifiers.
    model = _schema_fb.ModelT.InitFromObj(
        _schema_fb.Model.GetRootAsModel(model_buf, 0))
    buffer_field = _schema_fb.BufferT()
    buffer_field.data = metadata_buf
    model.buffers = [buffer_field]
    # Creates a new metadata field.
    metadata_field = _schema_fb.MetadataT()
    metadata_field.name = _metadata.MetadataPopulator.METADATA_FIELD_NAME
    metadata_field.buffer = len(model.buffers) - 1
    model.metadata = [metadata_field]
    b = flatbuffers.Builder(0)
    b.Finish(model.Pack(b), identifier)
    return b.Output()

  def _create_metadata_file_with_version(self, metadata_file, min_version):
    # Creates a new metadata file with the specified min_version for testing
    # purposes.
    with open(metadata_file, "rb") as f:
      metadata_buf = bytearray(f.read())

    metadata = _metadata_fb.ModelMetadataT.InitFromObj(
        _metadata_fb.ModelMetadata.GetRootAsModelMetadata(metadata_buf, 0))
    metadata.minParserVersion = min_version

    b = flatbuffers.Builder(0)
    b.Finish(
        metadata.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)

    metadata_file_with_version = self.create_tempfile().full_path
    with open(metadata_file_with_version, "wb") as f:
      f.write(b.Output())
    return metadata_file_with_version


class MetadataPopulatorTest(MetadataTest):

  def testToValidModelFile(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    self.assertIsInstance(populator, _metadata.MetadataPopulator)

  def testToInvalidModelFile(self):
    with self.assertRaises(IOError) as error:
      _metadata.MetadataPopulator.with_model_file(self._invalid_file)
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testToValidModelBuffer(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    self.assertIsInstance(populator, _metadata.MetadataPopulator)

  def testToInvalidModelBuffer(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataPopulator.with_model_buffer(self._invalid_model_buf)
    self.assertEqual("model_buf cannot be empty.", str(error.exception))

  def testToModelBufferWithWrongIdentifier(self):
    model_buf = self._create_model_buffer_with_wrong_identifier()
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataPopulator.with_model_buffer(model_buf)
    self.assertEqual(
        "The model provided does not have the expected identifier, and "
        "may not be a valid TFLite model.", str(error.exception))

  def testSinglePopulateAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    populator.load_associated_files([self._file1])
    populator.populate()

    packed_files = populator.get_packed_associated_file_list()
    expected_packed_files = [os.path.basename(self._file1)]
    self.assertEqual(set(packed_files), set(expected_packed_files))

  def testRepeatedPopulateAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_associated_files([self._file1, self._file2])
    # Loads file2 multiple times.
    populator.load_associated_files([self._file2])
    populator.populate()

    packed_files = populator.get_packed_associated_file_list()
    expected_packed_files = [
        os.path.basename(self._file1),
        os.path.basename(self._file2)
    ]
    self.assertEqual(len(packed_files), 2)
    self.assertEqual(set(packed_files), set(expected_packed_files))

    # Check if the model buffer read from file is the same as that read from
    # get_model_buffer().
    with open(self._model_file, "rb") as f:
      model_buf_from_file = f.read()
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testPopulateInvalidAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(IOError) as error:
      populator.load_associated_files([self._invalid_file])
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testPopulatePackedAssociatedFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    populator.load_associated_files([self._file1])
    populator.populate()
    with self.assertRaises(ValueError) as error:
      populator.load_associated_files([self._file1])
      populator.populate()
    self.assertEqual(
        "File, '{0}', has already been packed.".format(
            os.path.basename(self._file1)), str(error.exception))

  def testGetPackedAssociatedFileList(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    packed_files = populator.get_packed_associated_file_list()
    self.assertEqual(packed_files, [])

  def testPopulateMetadataFileToEmptyModelFile(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1, self._file2])
    populator.populate()

    with open(self._model_file, "rb") as f:
      model_buf_from_file = f.read()
    model = _schema_fb.Model.GetRootAsModel(model_buf_from_file, 0)
    # self._model_file already has two elements in the metadata field, so the
    # populated TFLite metadata will be the third element.
    metadata_field = model.Metadata(2)
    self.assertEqual(
        six.ensure_str(metadata_field.Name()),
        six.ensure_str(_metadata.MetadataPopulator.METADATA_FIELD_NAME))

    buffer_index = metadata_field.Buffer()
    buffer_data = model.Buffers(buffer_index)
    metadata_buf_np = buffer_data.DataAsNumpy()
    metadata_buf = metadata_buf_np.tobytes()
    with open(self._metadata_file_with_version, "rb") as f:
      expected_metadata_buf = bytearray(f.read())
    self.assertEqual(metadata_buf, expected_metadata_buf)

    recorded_files = populator.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set(self.expected_recorded_files))

    # Up to now, we've proved the correctness of the model buffer that read from
    # file. Then we'll test if get_model_buffer() gives the same model buffer.
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testPopulateMetadataFileWithoutAssociatedFiles(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1])
    # Suppose to populate self._file2, because it is recorded in the metadta.
    with self.assertRaises(ValueError) as error:
      populator.populate()
    self.assertEqual(("File, '{0}', is recorded in the metadata, but has "
                      "not been loaded into the populator.").format(
                          os.path.basename(self._file2)), str(error.exception))

  def testPopulateMetadataBufferWithWrongIdentifier(self):
    metadata_buf = self._create_metadata_buffer_with_wrong_identifier()
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer(metadata_buf)
    self.assertEqual(
        "The metadata buffer does not have the expected identifier, and may not"
        " be a valid TFLite Metadata.", str(error.exception))

  def _assert_golden_metadata(self, model_file):
    with open(model_file, "rb") as f:
      model_buf_from_file = f.read()
    model = _schema_fb.Model.GetRootAsModel(model_buf_from_file, 0)
    # There are two elements in model.Metadata array before the population.
    # Metadata should be packed to the third element in the array.
    metadata_field = model.Metadata(2)
    self.assertEqual(
        six.ensure_str(metadata_field.Name()),
        six.ensure_str(_metadata.MetadataPopulator.METADATA_FIELD_NAME))

    buffer_index = metadata_field.Buffer()
    buffer_data = model.Buffers(buffer_index)
    metadata_buf_np = buffer_data.DataAsNumpy()
    metadata_buf = metadata_buf_np.tobytes()
    with open(self._metadata_file_with_version, "rb") as f:
      expected_metadata_buf = bytearray(f.read())
    self.assertEqual(metadata_buf, expected_metadata_buf)

  def testPopulateMetadataFileToModelWithMetadataAndAssociatedFiles(self):
    # First, creates a dummy metadata different from self._metadata_file. It
    # needs to have the same input/output tensor numbers as self._model_file.
    # Populates it and the associated files into the model.
    input_meta = _metadata_fb.TensorMetadataT()
    output_meta = _metadata_fb.TensorMetadataT()
    subgraph = _metadata_fb.SubGraphMetadataT()
    # Create a model with two inputs and one output.
    subgraph.inputTensorMetadata = [input_meta, input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.subgraphMetadata = [subgraph]
    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    # Populate the metadata.
    populator1 = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator1.load_metadata_buffer(metadata_buf)
    populator1.load_associated_files([self._file1, self._file2])
    populator1.populate()

    # Then, populate the metadata again.
    populator2 = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator2.load_metadata_file(self._metadata_file)
    populator2.populate()

    # Test if the metadata is populated correctly.
    self._assert_golden_metadata(self._model_file)

  def testPopulateMetadataFileToModelFileWithMetadataAndBufFields(self):
    populator = _metadata.MetadataPopulator.with_model_file(self._model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1, self._file2])
    populator.populate()

    # Tests if the metadata is populated correctly.
    self._assert_golden_metadata(self._model_file)

    recorded_files = populator.get_recorded_associated_file_list()
    self.assertEqual(set(recorded_files), set(self.expected_recorded_files))

    # Up to now, we've proved the correctness of the model buffer that read from
    # file. Then we'll test if get_model_buffer() gives the same model buffer.
    with open(self._model_file, "rb") as f:
      model_buf_from_file = f.read()
    model_buf_from_getter = populator.get_model_buffer()
    self.assertEqual(model_buf_from_file, model_buf_from_getter)

  def testPopulateInvalidMetadataFile(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(IOError) as error:
      populator.load_metadata_file(self._invalid_file)
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def testPopulateInvalidMetadataBuffer(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer([])
    self.assertEqual("The metadata to be populated is empty.",
                     str(error.exception))

  def testGetModelBufferBeforePopulatingData(self):
    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    model_buf = populator.get_model_buffer()
    expected_model_buf = self._model_buf
    self.assertEqual(model_buf, expected_model_buf)

  def testLoadMetadataBufferWithNoSubgraphMetadataThrowsException(self):
    # Create a dummy metadata without Subgraph.
    model_meta = _metadata_fb.ModelMetadataT()
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    meta_buf = builder.Output()

    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer(meta_buf)
    self.assertEqual(
        "The number of SubgraphMetadata should be exactly one, but got 0.",
        str(error.exception))

  def testLoadMetadataBufferWithWrongInputMetaNumberThrowsException(self):
    # Create a dummy metadata with no input tensor metadata, while the expected
    # number is 2.
    output_meta = _metadata_fb.TensorMetadataT()
    subgprah_meta = _metadata_fb.SubGraphMetadataT()
    subgprah_meta.outputTensorMetadata = [output_meta]
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.subgraphMetadata = [subgprah_meta]
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    meta_buf = builder.Output()

    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer(meta_buf)
    self.assertEqual(
        ("The number of input tensors (2) should match the number of "
         "input tensor metadata (0)"), str(error.exception))

  def testLoadMetadataBufferWithWrongOutputMetaNumberThrowsException(self):
    # Create a dummy metadata with no output tensor metadata, while the expected
    # number is 1.
    input_meta = _metadata_fb.TensorMetadataT()
    subgprah_meta = _metadata_fb.SubGraphMetadataT()
    subgprah_meta.inputTensorMetadata = [input_meta, input_meta]
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.subgraphMetadata = [subgprah_meta]
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    meta_buf = builder.Output()

    populator = _metadata.MetadataPopulator.with_model_buffer(self._model_buf)
    with self.assertRaises(ValueError) as error:
      populator.load_metadata_buffer(meta_buf)
    self.assertEqual(
        ("The number of output tensors (1) should match the number of "
         "output tensor metadata (0)"), str(error.exception))


class MetadataDisplayerTest(MetadataTest):

  def setUp(self):
    super(MetadataDisplayerTest, self).setUp()
    self._model_with_meta_file = (
        self._create_model_with_metadata_and_associated_files())

  def _create_model_with_metadata_and_associated_files(self):
    model_buf = self._create_model_buf()
    model_file = self.create_tempfile().full_path
    with open(model_file, "wb") as f:
      f.write(model_buf)

    populator = _metadata.MetadataPopulator.with_model_file(model_file)
    populator.load_metadata_file(self._metadata_file)
    populator.load_associated_files([self._file1, self._file2])
    populator.populate()
    return model_file

  def test_load_model_buffer_metadataBufferWithWrongIdentifier_throwsException(
      self):
    model_buf = self._create_model_buffer_with_wrong_identifier()
    metadata_buf = self._create_metadata_buffer_with_wrong_identifier()
    model_buf = self._populate_metadata_with_identifier(
        model_buf, metadata_buf,
        _metadata.MetadataPopulator.TFLITE_FILE_IDENTIFIER)
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_buffer(model_buf)
    self.assertEqual(
        "The metadata buffer does not have the expected identifier, and may not"
        " be a valid TFLite Metadata.", str(error.exception))

  def test_load_model_buffer_modelBufferWithWrongIdentifier_throwsException(
      self):
    model_buf = self._create_model_buffer_with_wrong_identifier()
    metadata_file = self._create_metadata_file()
    wrong_identifier = b"widn"
    with open(metadata_file, "rb") as f:
      metadata_buf = bytearray(f.read())
    model_buf = self._populate_metadata_with_identifier(model_buf, metadata_buf,
                                                        wrong_identifier)
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_buffer(model_buf)
    self.assertEqual(
        "The model provided does not have the expected identifier, and "
        "may not be a valid TFLite model.", str(error.exception))

  def test_load_model_file_invalidModelFile_throwsException(self):
    with self.assertRaises(IOError) as error:
      _metadata.MetadataDisplayer.with_model_file(self._invalid_file)
    self.assertEqual("File, '{0}', does not exist.".format(self._invalid_file),
                     str(error.exception))

  def test_load_model_file_modelWithoutMetadata_throwsException(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_file(self._model_file)
    self.assertEqual("The model does not have metadata.", str(error.exception))

  def test_load_model_file_modelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(
        self._model_with_meta_file)
    self.assertIsInstance(displayer, _metadata.MetadataDisplayer)

  def test_load_model_buffer_modelWithOutMetadata_throwsException(self):
    with self.assertRaises(ValueError) as error:
      _metadata.MetadataDisplayer.with_model_buffer(self._create_model_buf())
    self.assertEqual("The model does not have metadata.", str(error.exception))

  def test_load_model_buffer_modelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_buffer(
        open(self._model_with_meta_file, "rb").read())
    self.assertIsInstance(displayer, _metadata.MetadataDisplayer)

  def test_get_metadata_json_modelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(
        self._model_with_meta_file)
    actual = displayer.get_metadata_json()

    # Verifies the generated json file.
    golden_json_file_path = resource_loader.get_path_to_datafile(
        "testdata/golden_json.json")
    with open(golden_json_file_path, "r") as f:
      expected = f.read()
    self.assertEqual(actual, expected)

  def test_get_packed_associated_file_list_modelWithMetadata(self):
    displayer = _metadata.MetadataDisplayer.with_model_file(
        self._model_with_meta_file)
    packed_files = displayer.get_packed_associated_file_list()

    expected_packed_files = [
        os.path.basename(self._file1),
        os.path.basename(self._file2)
    ]
    self.assertEqual(len(packed_files), 2)
    self.assertEqual(set(packed_files), set(expected_packed_files))


if __name__ == "__main__":
  test.main()
