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
"""Utils for Task APIs."""

from tensorflow_lite_support.python.task.core.proto import configuration_pb2


def build_proto_base_options(proto_options, options):
    """Builds initial proto options values from the base options."""

    # Updates values from base_options.
    if options.base_options.model_file.file_content:
      proto_options.model_file_with_metadata.file_content = (
        options.base_options.model_file.file_content)
    elif options.base_options.model_file.file_name:
      proto_options.model_file_with_metadata.file_name = (
        options.base_options.model_file.file_name)

    proto_options.num_threads = options.base_options.num_threads
    if options.base_options.use_coral:
      proto_options.compute_settings.tflite_settings.delegate = (
        configuration_pb2.Delegate.EDGETPU_CORAL)

    return proto_options
