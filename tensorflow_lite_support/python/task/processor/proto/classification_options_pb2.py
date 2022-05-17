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
"""Classification options protobuf."""

import dataclasses
from typing import Any, List, Optional

from tensorflow.tools.docs import doc_controls
from tensorflow_lite_support.cc.task.processor.proto import classification_options_pb2

_ClassificationOptions = classification_options_pb2.ClassificationOptions


@dataclasses.dataclass
class ClassificationOptions:
  """Options for classification processor.

  Attributes:
    display_names_locale: The locale to use for display names specified through
      the TFLite Model Metadata.
    max_results: The maximum number of top-scored classification results to
      return.
    score_threshold: Overrides the ones provided in the model metadata. Results
      below this value are rejected.
    class_name_allowlist: If non-empty, classifications whose class name is not
      in this set will be filtered out. Duplicate or unknown class names are
      ignored. Mutually exclusive with `class_name_denylist`.
    class_name_denylist: If non-empty, classifications whose class name is in
      this set will be filtered out. Duplicate or unknown class names are
      ignored. Mutually exclusive with `class_name_allowlist`.
  """

  score_threshold: Optional[float] = None
  class_name_allowlist: Optional[List[str]] = None
  class_name_denylist: Optional[List[str]] = None
  display_names_locale: Optional[str] = None
  max_results: Optional[int] = None

  @doc_controls.do_not_generate_docs
  def to_pb2(self) -> _ClassificationOptions:
    """Generates a protobuf object to pass to the C++ layer."""
    return _ClassificationOptions(
      score_threshold=self.score_threshold,
      class_name_allowlist=self.class_name_allowlist,
      class_name_denylist=self.class_name_denylist,
      display_names_locale=self.display_names_locale,
      max_results=self.max_results)

  @classmethod
  @doc_controls.do_not_generate_docs
  def create_from_pb2(
      cls,
      pb2_obj: _ClassificationOptions = _ClassificationOptions()
  ) -> "ClassificationOptions":
    """Creates a `DetectionOptions` object from the given protobuf object."""
    return ClassificationOptions(
      score_threshold=pb2_obj.score_threshold,
      class_name_allowlist=pb2_obj.class_name_allowlist,
      class_name_denylist=pb2_obj.class_name_denylist,
      display_names_locale=pb2_obj.display_names_locale,
      max_results=pb2_obj.max_results)

  def __eq__(self, other: Any) -> bool:
    """Checks if this object is equal to the given object.

    Args:
      other: The object to be compared with.

    Returns:
      True if the objects are equal.
    """
    if not isinstance(other, ClassificationOptions):
      return False

    return self.to_pb2().__eq__(other.to_pb2())
