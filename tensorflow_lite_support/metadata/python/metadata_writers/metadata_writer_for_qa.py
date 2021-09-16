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
"""Write metadata for Descartes QA TFLite models."""
import pathlib

from absl import app
from absl import flags
import tensorflow as tf

from tensorflow_lite_support.metadata.python.metadata_writers import metadata_info
from tensorflow_lite_support.metadata.python.metadata_writers import metadata_writer
from tensorflow_lite_support.metadata.python.metadata_writers import writer_utils

FLAGS = flags.FLAGS


def define_flags():
  """Defines flags."""
  flags.DEFINE_string("model_file", None,
                      "Path and file name to the TFLite model file.")
  flags.DEFINE_string("export_file", None, "Path to exported model.")
  flags.mark_flag_as_required("model_file")
  flags.mark_flag_as_required("export_file")


class MetadataPopulatorForQA(object):
  """Populates the metadata for Descartes QA."""

  def __init__(self, model_file):
    self._model_file = model_file
    self._writer = self._create_metadata_writer()

  def _create_metadata_writer(self):
    """Creates the metadata writer."""
    general_md = metadata_info.GeneralMd(
        name="USE-QA TFLite model",
        version="v1",
        description="Greater-than-word length text encoder for question answer retrieval.",
        author="Google LLC")
    input_md = [
        metadata_info.TensorMd("inp_text", "Input query text"),
        metadata_info.TensorMd("res_context", "Response context"),
        metadata_info.TensorMd("res_text", "Response text"),
    ]
    output_md = [
        metadata_info.TensorMd(
            "query_encoding", "Query encoding for the dot-product similarity"),
        metadata_info.TensorMd(
            "response_encoding",
            "Response encoding for the dot-product similarity"),
    ]
    return metadata_writer.MetadataWriter.create_from_metadata_info(
        model_buffer=writer_utils.load_file(self._model_file),
        general_md=general_md,
        input_md=input_md,
        output_md=output_md)

  def populate(self, output_file):
    """Creates metadata and then populates it."""
    writer_utils.save_file(self._writer.populate(), output_file)

  def get_metadata_json(self):
    """Gets the generated metadata string in JSON format."""
    return self._writer.get_metadata_json()


def main(_):
  # Copies model_file to export_path.
  output_dir = pathlib.Path(FLAGS.export_file).parent
  output_dir.mkdir(exist_ok=True)

  # Generate the metadata objects and put them in the model file
  populator = MetadataPopulatorForQA(FLAGS.model_file)
  populator.populate(FLAGS.output_file)

  # Log as json format.
  logger = tf.get_logger()
  logger.info(populator.get_metadata_json())


if __name__ == "__main__":
  define_flags()
  app.run(main)
