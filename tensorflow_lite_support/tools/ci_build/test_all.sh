#!/usr/bin/env bash
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
# ==============================================================================
# External `test_all.sh`

# Due to resource limits, we don't always run these tests as presubmits. We will
# have continuous tests to monitor code health instead.

set -ex

source tensorflow_lite_support/tools/ci_build/tests/run_metadata_tests.sh
source tensorflow_lite_support/tools/ci_build/tests/run_support_lib_tests.sh

bazel test --test_output=all \
    //tensorflow_lite_support/cc/test/task/vision:all \
    //tensorflow_lite_support/custom_ops/kernel/sentencepiece:all
