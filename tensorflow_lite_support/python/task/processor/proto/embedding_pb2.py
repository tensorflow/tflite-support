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
"""Embedding result protobuf."""

import dataclasses
from typing import Any, List

from tensorflow.tools.docs import doc_controls
from tensorflow_lite_support.cc.task.processor.proto import embedding_pb2

_FeatureVector = embedding_pb2.FeatureVector
_Embedding = embedding_pb2.Embedding
_EmbeddingResult = embedding_pb2.EmbeddingResult


@dataclasses.dataclass
class FeatureVector:
  """A dense feature vector. Only one of the two fields is ever present.
  Feature vectors are assumed to be one-dimensional and L2-normalized.

  Attributes:
    value_float: Raw output of the embedding layer. Only provided if `quantize`
      is set to False in the `EmbeddingOptions`, which is the case by default.
    value_string: Scalar-quantized embedding. Only provided if `quantize` is
      set to true in `EmbeddingOptions`.
  """

  value_float: List[float] = None
  value_string: bytearray = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _FeatureVector:
    """Generates a protobuf object to pass to the C++ layer."""

    if self.value_float is not None:
      return _FeatureVector(value_float=self.value_float)

    elif self.value_string is not None:
      return _FeatureVector(value_string=bytes(self.value_string))

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _FeatureVector) -> "FeatureVector":
    """Creates a `FeatureVector` object from the given protobuf object."""

    if pb2_obj.value_float:
      return FeatureVector(value_float=pb2_obj.value_float)

    elif pb2_obj.value_string:
      return FeatureVector(value_string=bytearray(pb2_obj.value_string))

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, FeatureVector):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class Embedding:
  """Result produced by one of the embedder model output layers.

  Attributes:
    feature_vector: The output feature vector.
    output_index: The index of the model output layer that produced this
      feature vector.
  """

  feature_vector: FeatureVector
  output_index: int

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _Embedding:
    """Generates a protobuf object to pass to the C++ layer."""
    return _Embedding(feature_vector=self.feature_vector.to_pb2(),
                      output_index=self.output_index)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(cls, pb2_obj: _Embedding) -> "Embedding":
    """Creates a `Embedding` object from the given protobuf object."""
    return Embedding(
      feature_vector=FeatureVector.create_from_pb2(
        pb2_obj.feature_vector),
      output_index=pb2_obj.output_index)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, Embedding):
      return False

    return self.to_pb2().__eq__(other.to_pb2())


@dataclasses.dataclass
class EmbeddingResult:
  """Embeddings produced by the Embedder.

  Attributes:
    embeddings: The embeddings produced by each of the model output layers.
      Except in advanced cases, the embedding model has a single output layer,
      and this list is thus made of a single element feature vector.
  """

  embeddings: List[Embedding]

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _EmbeddingResult:
    """Generates a protobuf object to pass to the C++ layer."""
    return _EmbeddingResult(
      embeddings=[embedding.to_pb2() for embedding in self.embeddings])

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls,
      pb2_obj: _EmbeddingResult
  ) -> "EmbeddingResult":
    """Creates a `EmbeddingResult` object from the given protobuf object."""
    return EmbeddingResult(
      embeddings=[
        Embedding.create_from_pb2(embedding)
        for embedding in pb2_obj.embeddings])

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.
    Args:
      other: The object to be compared with.
    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, EmbeddingResult):
      return False
