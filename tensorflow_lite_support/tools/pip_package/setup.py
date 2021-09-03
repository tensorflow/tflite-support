# lint as: python3
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# limitations under the License..
# ==============================================================================
"""TFLite Support is a toolkit that helps users to develop ML and deploy TFLite models onto mobile devices.

This PyPI package includes the Python bindings for following features:

 - Metadata schemas: wraps TFLite model schema and metadata schema in Python.
 - Metadata populator and displayer: can be used to populate the metadata and
 associated files into the model, as well as converting the populated metadata
 into the json format.
 - Android Codegen tool: generates the Java model interface used in Android for
 a particular model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import re
import sys

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
_VERSION = '0.1.0'

SETUP_PACKAGES = [
    'pybind11 >= 2.6.0',
]

REQUIRED_PACKAGES = [
    'absl-py >= 0.7.0',
    'numpy >= 1.19.2',
    'flatbuffers >= 2.0.0',
] + SETUP_PACKAGES

project_name = 'tflite-support'
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)

DOCLINES = __doc__.split('\n')

CONSOLE_SCRIPTS = [
    'tflite_codegen = tensorflow_lite_support.codegen.python.codegen:main',
]


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)
    self.install_lib = self.install_platlib
    return ret


def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

matches = []
for path in so_lib_paths:
  matches.extend(['../' + x for x in find_files('*', path) if '.py' not in x])

EXTENSIONS = ['codegen/_pywrap_codegen.so']

headers = ()

setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES),
    long_description_content_type='text/markdown',
    url='https://www.tensorflow.org/',
    download_url='https://github.com/tensorflow/tflite-support/tags',
    author='Google, LLC.',
    author_email='packages@tensorflow.org',
    # Contained modules and scripts.
    packages=find_packages(),
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    headers=headers,
    setup_requires=SETUP_PACKAGES,
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    package_data={
        'tflite-support': EXTENSIONS + matches,
    },
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'install': InstallCommand,
    },
    # PyPI package information.
    classifiers=sorted([
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]),
    license='Apache 2.0',
)
