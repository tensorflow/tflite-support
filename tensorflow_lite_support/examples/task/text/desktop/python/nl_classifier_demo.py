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
import subprocess
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('model_path', None, 'Model Path')
flags.DEFINE_string('text', None, 'Text to Predict')

# Required flag.
flags.mark_flag_as_required('model_path')
flags.mark_flag_as_required('text')


def classify(model_path, text):
  """Classifies input text into different categories

  Args:
      model_path: path to model
      text: input text

  """
  # Run the detection tool:
  subprocess.run([
      'bazel run -c opt  '
      'tensorflow_lite_support/examples/task/text/desktop:nl_classifier_demo --  '
      '--model_path="' + model_path + '"  --text="' + text + '"'
  ],
                 shell=True,
                 check=True)


def main(argv):
  del argv  # Unused.
  classify(FLAGS.model_path, FLAGS.text)


if __name__ == '__main__':
  app.run(main)
