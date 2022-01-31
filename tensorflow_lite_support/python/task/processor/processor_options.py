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
"""Options to configure processors."""

import dataclasses
from typing import Optional, List


@dataclasses.dataclass
class EmbeddingOptions:
  """Options for embedding processor.
  Attributes:
    l2_normalize: Whether to normalize the returned feature vector with L2 norm.
      Use this option only if the model does not already contain a native
      L2_NORMALIZATION TF Lite Op. In most cases, this is already the case and
      L2 norm is thus achieved through TF Lite inference.
    quantize: Whether the returned embedding should be quantized to bytes via
      scalar quantization. Embeddings are implicitly assumed to be unit-norm and
      therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
      the l2_normalize option if this is not the case.
  """

  l2_normalize: Optional[bool] = None

  quantize: Optional[bool] = None


@dataclasses.dataclass
class ClassificationOptions:
  """Options for classification processor.
  Attributes:
    display_names_locale: The locale to use for display names specified
      through the TFLite Model Metadata, if any. Defaults to English.
    max_results: The maximum number of top-scored classification results to
      return. If < 0, all available results will be returned. If 0, an
      invalid argument error is returned.
    score_threshold: Score threshold, overrides the ones provided in the
      model metadata (if any). Results below this value are rejected. It
      is tempting to assume that the classification threshold should always
      be 0.5, but thresholds are problem-dependent, and are therefore values
      that you must tune.
    label_allowlist: Optional allowlist of class names. If non-empty,
      classifications whose class name is not in this set will be filtered out.
      Duplicate or unknown class names are ignored. Mutually exclusive with
      label_denylist.
    label_denylist: Optional denylist of class names. If non-empty,
      classifications whose class name is in this set will be filtered out.
      Duplicate or unknown class names are ignored. Mutually exclusive with
      label_allowlist.
  """

  display_names_locale: Optional[str] = None

  max_results: Optional[int] = None

  score_threshold: Optional[float] = None

  label_allowlist: List[str] = None

  label_denylist: List[str] = None
