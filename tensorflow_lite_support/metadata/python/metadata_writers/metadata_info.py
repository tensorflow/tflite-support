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
"""Helper classes for common model metadata information."""

import os
from typing import Optional, List, Type

from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
from tensorflow_lite_support.metadata import schema_py_generated as _schema_fb

# Min and max values for UINT8 tensors.
_MIN_UINT8 = 0
_MAX_UINT8 = 255


class GeneralMd:
  """A container for common metadata information of a model.

  Attributes:
    name: name of the model.
    version: version of the model.
    description: description of what the model does.
    author: author of the model.
    licenses: licenses of the model.
  """

  def __init__(self,
               name: Optional[str] = None,
               version: Optional[str] = None,
               description: Optional[str] = None,
               author: Optional[str] = None,
               licenses: Optional[str] = None):
    self.name = name
    self.version = version
    self.description = description
    self.author = author
    self.licenses = licenses

  def create_metadata(self) -> _metadata_fb.ModelMetadataT:
    """Creates the model metadata based on the general model information.

    Returns:
      A Flatbuffers Python object of the model metadata.
    """
    model_metadata = _metadata_fb.ModelMetadataT()
    model_metadata.name = self.name
    model_metadata.version = self.version
    model_metadata.description = self.description
    model_metadata.author = self.author
    model_metadata.license = self.licenses
    return model_metadata


class AssociatedFileMd:
  """A container for common associated file metadata information.

  Attributes:
    file_path: path to the associated file.
    description: description of the associated file.
    file_type: file type of the associated file [1].
    locale: locale of the associated file [2].
    [1]:
      https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L77
    [2]:
      https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L154
  """

  def __init__(
      self,
      file_path: Optional[str] = None,
      description: Optional[str] = None,
      file_type: Optional[_metadata_fb.AssociatedFileType] = _metadata_fb
      .AssociatedFileType.UNKNOWN,
      locale: Optional[str] = None):
    self.file_path = file_path
    self.description = description
    self.file_type = file_type
    self.locale = locale

  def create_metadata(self) -> _metadata_fb.AssociatedFileT:
    """Creates the associated file metadata.

    Returns:
      A Flatbuffers Python object of the associated file metadata.
    """
    file_metadata = _metadata_fb.AssociatedFileT()
    file_metadata.name = os.path.basename(self.file_path)
    file_metadata.description = self.description
    file_metadata.type = self.file_type
    file_metadata.locale = self.locale
    return file_metadata


class LabelFileMd(AssociatedFileMd):
  """A container for label file metadata information."""

  _LABEL_FILE_DESCRIPTION = ("Labels for categories that the model can "
                             "recognize.")
  _FILE_TYPE = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS

  def __init__(self,
               file_path: Optional[str] = None,
               locale: Optional[str] = None):
    """Creates a LabelFileMd object.

    Args:
      file_path: file_path of the label file.
      locale: locale of the label file [1].
      [1]:
      https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L154
    """
    super().__init__(file_path, self._LABEL_FILE_DESCRIPTION, self._FILE_TYPE,
                     locale)


class TensorMd:
  """A container for common tensor metadata information.

  Attributes:
    name: name of the tensor.
    description: description of what the tensor is.
    min_values: per-channel minimum value of the tensor.
    max_values: per-channel maximum value of the tensor.
    content_type: content_type of the tensor.
    associated_files: information of the associated files in the tensor.
  """

  def __init__(self,
               name: Optional[str] = None,
               description: Optional[str] = None,
               min_values: Optional[List[float]] = None,
               max_values: Optional[List[float]] = None,
               content_type: _metadata_fb.ContentProperties = _metadata_fb
               .ContentProperties.FeatureProperties,
               associated_files: Optional[List[Type[AssociatedFileMd]]] = None):
    self.name = name
    self.description = description
    self.min_values = min_values
    self.max_values = max_values
    self.content_type = content_type
    self.associated_files = associated_files

  def create_metadata(self) -> _metadata_fb.TensorMetadataT:
    """Creates the input tensor metadata based on the information.

    Returns:
      A Flatbuffers Python object of the input metadata.
    """
    tensor_metadata = _metadata_fb.TensorMetadataT()
    tensor_metadata.name = self.name
    tensor_metadata.description = self.description

    # Create min and max values
    stats = _metadata_fb.StatsT()
    stats.max = self.max_values
    stats.min = self.min_values
    tensor_metadata.stats = stats

    # Create content properties
    content = _metadata_fb.ContentT()
    if self.content_type is _metadata_fb.ContentProperties.FeatureProperties:
      content.contentProperties = _metadata_fb.FeaturePropertiesT()
    elif self.content_type is _metadata_fb.ContentProperties.ImageProperties:
      content.contentProperties = _metadata_fb.ImagePropertiesT()
    elif self.content_type is (
        _metadata_fb.ContentProperties.BoundingBoxProperties):
      content.contentProperties = _metadata_fb.BoundingBoxPropertiesT()
    content.contentPropertiesType = self.content_type
    tensor_metadata.content = content

    # Create associated files
    if self.associated_files:
      tensor_metadata.associatedFiles = [
          file.create_metadata() for file in self.associated_files
      ]
    return tensor_metadata


class InputImageTensorMd(TensorMd):
  """A container for input tensor metadata information.

  Attributes:
    norm_mean: the mean value used in tensor normalization [1].
    norm_std: the std value used in the tensor normalization [1]. norm_mean and
      norm_std must have the same dimension.
    color_space_type: the color space type of the input image [2].
    [1]:
      https://www.tensorflow.org/lite/convert/metadata#normalization_and_quantization_parameters
    [2]:
      https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L172
  """

  # Min and max float values for image pixels.
  _MIN_PIXEL = 0.0
  _MAX_PIXEL = 255.0

  def __init__(
      self,
      name: Optional[str] = None,
      description: Optional[str] = None,
      norm_mean: Optional[List[float]] = None,
      norm_std: Optional[List[float]] = None,
      color_space_type: Optional[
          _metadata_fb.ColorSpaceType] = _metadata_fb.ColorSpaceType.UNKNOWN,
      tensor_type: Optional[_schema_fb.TensorType] = None):
    """Initializes the instance of InputImageTensorMd.

    Args:
      name: name of the tensor.
      description: description of what the tensor is.
      norm_mean: the mean value used in tensor normalization [1].
      norm_std: the std value used in the tensor normalization [1]. norm_mean
        and norm_std must have the same dimension.
      color_space_type: the color space type of the input image [2].
      tensor_type: data type of the tensor.
      [1]:
        https://www.tensorflow.org/lite/convert/metadata#normalization_and_quantization_parameters
      [2]:
      https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L172

    Raises:
      ValueError: if norm_mean and norm_std have different dimensions.
    """
    if norm_std and norm_mean and len(norm_std) != len(norm_mean):
      # TODO(b/175843689): Python version cannot be specified in Kokoro bazel
      # test
      raise ValueError(
          "norm_mean and norm_std are expected to be the same dim. But got " +
          "{} and {}".format(len(norm_mean), len(norm_std)))

    if tensor_type is _schema_fb.TensorType.UINT8:
      min_values = [_MIN_UINT8]
      max_values = [_MAX_UINT8]
    elif tensor_type is _schema_fb.TensorType.FLOAT32 and norm_std and norm_mean:
      min_values = [
          float(self._MIN_PIXEL - mean) / std
          for mean, std in zip(norm_mean, norm_std)
      ]
      max_values = [
          float(self._MAX_PIXEL - mean) / std
          for mean, std in zip(norm_mean, norm_std)
      ]
    else:
      # Uint8 and Float32 are the two major types currently. And Task library
      # doesn't support other types so far.
      min_values = None
      max_values = None

    super().__init__(name, description, min_values, max_values,
                     _metadata_fb.ContentProperties.ImageProperties)
    self.norm_mean = norm_mean
    self.norm_std = norm_std
    self.color_space_type = color_space_type

  def create_metadata(self) -> _metadata_fb.TensorMetadataT:
    """Creates the input image metadata based on the information.

    Returns:
      A Flatbuffers Python object of the input image metadata.
    """
    tensor_metadata = super().create_metadata()
    tensor_metadata.content.contentProperties.colorSpace = self.color_space_type
    # Create normalization parameters
    if self.norm_mean and self.norm_std:
      normalization = _metadata_fb.ProcessUnitT()
      normalization.optionsType = (
          _metadata_fb.ProcessUnitOptions.NormalizationOptions)
      normalization.options = _metadata_fb.NormalizationOptionsT()
      normalization.options.mean = self.norm_mean
      normalization.options.std = self.norm_std
      tensor_metadata.processUnits = [normalization]
    return tensor_metadata


class ClassificationTensorMd(TensorMd):
  """A container for the classification tensor metadata information.

  Attributes:
    label_files: information of the label files [1] in the classification
      tensor.
    [1]:
      https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L95
  """

  # Min and max float values for classification results.
  _MIN_FLOAT = 0.0
  _MAX_FLOAT = 1.0

  def __init__(self,
               name: Optional[str] = None,
               description: Optional[str] = None,
               label_files: Optional[List[LabelFileMd]] = None,
               tensor_type: Optional[_schema_fb.TensorType] = None):
    """Initializes the instance of ClassificationTensorMd.

    Args:
      name: name of the tensor.
      description: description of what the tensor is.
      label_files: information of the label files [1] in the classification
        tensor.
      tensor_type: data type of the tensor.
      [1]:
        https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L95
    """
    if tensor_type is _schema_fb.TensorType.UINT8:
      min_values = [_MIN_UINT8]
      max_values = [_MAX_UINT8]
    elif tensor_type is _schema_fb.TensorType.FLOAT32:
      min_values = [self._MIN_FLOAT]
      max_values = [self._MAX_FLOAT]
    else:
      # Uint8 and Float32 are the two major types currently. And Task library
      # doesn't support other types so far.
      min_values = None
      max_values = None

    super().__init__(name, description, min_values, max_values,
                     _metadata_fb.ContentProperties.FeatureProperties,
                     label_files)


class CategoryTensorMd(TensorMd):
  """A container for the category tensor metadata information."""

  def __init__(self,
               name: Optional[str] = None,
               description: Optional[str] = None,
               label_files: Optional[List[LabelFileMd]] = None):
    """Initializes a CategoryTensorMd object.

    Args:
      name: name of the tensor.
      description: description of what the tensor is.
      label_files: information of the label files [1] in the category tensor.
      [1]:
        https://github.com/tensorflow/tflite-support/blob/b80289c4cd1224d0e1836c7654e82f070f9eefaa/tensorflow_lite_support/metadata/metadata_schema.fbs#L108
    """
    # In category tensors, label files are in the type of TENSOR_VALUE_LABELS.
    value_label_files = label_files
    if value_label_files:
      for file in value_label_files:
        file.file_type = _metadata_fb.AssociatedFileType.TENSOR_VALUE_LABELS

    super().__init__(
        name=name, description=description, associated_files=value_label_files)
