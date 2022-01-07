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
"""Options to configure Task APIs."""

import dataclasses


@dataclasses.dataclass
class BaseOptions:
  """Base options that is used for creation of any type of task."""
  # Path to the model.
  model_file: str
  # Number of thread: the defaule value -1 means Interpreter will decide what
  # is the most appropriate num_threads.
  num_threads: int = -1
  # If true, inference will be delegated to a connected Coral Edge TPU device.
  use_coral: bool = False
