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
"""Options to configure Embedding APIs."""

import dataclasses
from typing import Optional, List


@dataclasses.dataclass
class EmbeddingOptions:
  """Options for embedding processor."""
  # Whether to normalize the returned feature vector with L2 norm. Use this
  # option only if the model does not already contain a native L2_NORMALIZATION
  # TF Lite Op. In most cases, this is already the case and L2 norm is thus
  # achieved through TF Lite inference.
  l2_normalize: Optional[bool] = None

  # Whether the returned embedding should be quantized to bytes via scalar
  # quantization. Embeddings are implicitly assumed to be unit-norm and
  # therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
  # the l2_normalize option if this is not the case.
  quantize: Optional[bool] = None


@dataclasses.dataclass
class ClassificationOptions:
  """Options for classification processor."""
  # TODO: document

  display_names_locale: Optional[str] = None

  max_results: Optional[int] = None

  score_threshold: Optional[float] = None

  class_name_whitelist: List[str] = None

  class_name_blacklist: List[str] = None
