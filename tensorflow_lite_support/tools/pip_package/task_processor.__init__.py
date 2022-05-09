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
# ==============================================================================
"""TensorFlow Lite Task Library's processor module.

This module contains classes related to the pre-processing and post-processing
steps of the Task Library.
"""

from tensorflow_lite_support.python.task.processor.proto import bounding_box_pb2
from tensorflow_lite_support.python.task.processor.proto import classification_options_pb2
from tensorflow_lite_support.python.task.processor.proto import classifications_pb2
from tensorflow_lite_support.python.task.processor.proto import detection_options_pb2
from tensorflow_lite_support.python.task.processor.proto import detections_pb2
from tensorflow_lite_support.python.task.processor.proto import embedding_options_pb2
from tensorflow_lite_support.python.task.processor.proto import embedding_pb2
from tensorflow_lite_support.python.task.processor.proto import segmentation_options_pb2
from tensorflow_lite_support.python.task.processor.proto import segmentations_pb2
from tensorflow_lite_support.python.task.processor.proto import search_options_pb2
from tensorflow_lite_support.python.task.processor.proto import search_result_pb2

BoundingBox = bounding_box_pb2.BoundingBox
ClassificationOptions = classification_options_pb2.ClassificationOptions
Classifications = classifications_pb2.Classifications
DetectionOptions = detection_options_pb2.DetectionOptions
DetectionResult = detections_pb2.DetectionResult
EmbeddingOptions = embedding_options_pb2.EmbeddingOptions
Embedding = embedding_pb2.Embedding
SegmentationOptions = segmentation_options_pb2.SegmentationOptions
Segmentation = segmentations_pb2.Segmentation
SearchOptions = search_options_pb2.SearchOptions
SearchResult = search_result_pb2.SearchResult

# Remove unnecessary modules to avoid duplication in API docs.
del bounding_box_pb2
del classification_options_pb2
del classifications_pb2
del detection_options_pb2
del detections_pb2
del embedding_options_pb2
del embedding_pb2
del segmentation_options_pb2
del segmentations_pb2
del search_options_pb2
del search_result_pb2
