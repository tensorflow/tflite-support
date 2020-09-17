:: Copyright 2019 The TensorFlow Authors. All Rights Reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
:: =============================================================================

:: This script is shamefully borrowed from:
:: //third_party/tensorflow/tools/ci_build/release/common_win.bat.oss

echo on

@REM
@REM Set Environment Variables
@REM
SET PY_EXE=C:\%PYTHON_DIRECTORY%\python.exe
SET PIP_EXE=C:\%PYTHON_DIRECTORY%\Scripts\pip.exe
SET PATH=C:\%PYTHON_DIRECTORY%;C:\%PYTHON_DIRECTORY%\Scripts;%PATH%

%PIP_EXE% install flatbuffers==1.12 --upgrade --no-deps
%PIP_EXE% install setuptools --upgrade
%PIP_EXE% install numpy==1.16.0 --upgrade --no-deps

@REM
@REM Setup Bazel
@REM
:: Download Bazel from github and make sure its found in PATH.
SET BAZEL_VERSION=3.1.0
md C:\tools\bazel\
wget -q https://github.com/bazelbuild/bazel/releases/download/%BAZEL_VERSION%/bazel-%BAZEL_VERSION%-windows-x86_64.exe -O C:/tools/bazel/bazel.exe
SET PATH=C:\tools\bazel;%PATH%
bazel version
