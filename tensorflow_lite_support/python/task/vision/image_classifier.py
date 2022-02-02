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
"""Image classifier task."""

import dataclasses
from typing import Optional

from tensorflow_lite_support.python.task.core import task_options
from tensorflow_lite_support.python.task.processor import processor_options
from tensorflow_lite_support.python.task.vision.proto import image_classifier_options_pb2

from tensorflow_lite_support.python.task.core.utils import build_proto_base_options


@dataclasses.dataclass
class ImageClassifierOptions:
    """Options for the image classifier task."""
    base_options: task_options.BaseOptions
    classifier_options: Optional[processor_options.ClassificationOptions] = None


def _build_proto_options(
        options: ImageClassifierOptions
) -> image_classifier_options_pb2.ClassifierOptions:
    """Builds the protobuf image classifier options."""
    # Builds the initial proto_options.
    proto_options = image_classifier_options_pb2.ClassifierOptions()

    # Updates values from base_options.
    proto_options = build_proto_base_options(proto_options, options)

    # Updates values from classifier_options.
    if options.classifier_options:
        if options.classifier_options.display_names_locale is not None:
            proto_options.display_names_locale = options.classifier_options.display_names_locale
        if options.classifier_options.max_results is not None:
            proto_options.max_results = options.classifier_options.max_results
        if options.classifier_options.score_threshold is not None:
            proto_options.score_threshold = options.classifier_options.score_threshold
        if options.classifier_options.label_allowlist is not None:
            proto_options.class_name_whitelist.extend(options.classifier_options.label_allowlist)
        if options.classifier_options.label_denylist is not None:
            proto_options.class_name_blacklist.extend(options.classifier_options.label_denylist)

    return proto_options
