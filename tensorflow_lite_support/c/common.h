/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_SUPPORT_C_COMMON_H_
#define TENSORFLOW_LITE_SUPPORT_C_COMMON_H_

// Defines C struct and error codes for describing any error returned from the C
// Task Library.

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Error codes for TensorFlow Lite Task Library C APIs.
//
// One to one mapping with `TfLiteSupportStatus` code starting from kError = 1.
// Omits `kOk` since `TfLiteErrorCode` is only to be used in the event of an
// error and does not account for success unlike `TfLiteSupportStatus`. In case
// of success, TensorFlow Lite Task Library C APIs return the appropriate return
// value and a null error. One to one mapping makes it easier to convert between
// `TfLiteSupportStatus` and `TfLiteSupportErrorCode` without long switch statements.
// kErrorCodeFirst and kErrorCodeLast are also provided for safety checks during
// conversion. In case of modifications in error codes, ensure that
// kErrorCodeFirst is set to the least enum value and kErrorCodeLast is set to
// the greatest enum value.
enum TfLiteSupportErrorCode {
  // Unspecified error.
  kError = 1,
  // Invalid argument specified.
  kInvalidArgumentError = 2,
  // Invalid FlatBuffer file or buffer specified.
  kInvalidFlatBufferError = 3,
  // Model contains a builtin op that isn't supported by the OpResolver or
  // delegates.
  kUnsupportedBuiltinOpError = 4,
  // Model contains a custom op that isn't supported by the OpResolver or
  // delegates.
  kUnsupportedCustomOpError = 5,

  // File I/O error codes.

  // No such file.
  kFileNotFoundError = 100,
  // Permission issue.
  kFilePermissionDeniedError,
  // I/O error when reading file.
  kFileReadError,
  // I/O error when mmap-ing file.
  kFileMmapError,

  // TensorFlow Lite metadata error codes.

  // Unexpected schema version (aka file_identifier) in the Metadata FlatBuffer.
  kMetadataInvalidSchemaVersionError = 200,
  // No such associated file within metadata, or file has not been packed.
  kMetadataAssociatedFileNotFoundError,
  // ZIP I/O error when unpacking an associated file.
  kMetadataAssociatedFileZipError,
  // Inconsistency error between the metadata and actual TF Lite model.
  // E.g.: number of labels and output tensor values differ.
  kMetadataInconsistencyError,
  // Invalid process units specified.
  // E.g.: multiple ProcessUnits with the same type for a given tensor.
  kMetadataInvalidProcessUnitsError,
  // Inconsistency error with the number of labels.
  // E.g.: label files for different locales have a different number of labels.
  kMetadataNumLabelsMismatchError,
  // Score calibration parameters parsing error.
  // E.g.: too many parameters provided in the corresponding associated file.
  kMetadataMalformedScoreCalibrationError,
  // Unexpected number of subgraphs for the current task.
  // E.g.: image classification expects a single subgraph.
  kMetadataInvalidNumSubgraphsError,
  // A given tensor requires NormalizationOptions but none were found.
  // E.g.: float input tensor requires normalization to preprocess input images.
  kMetadataMissingNormalizationOptionsError,
  // Invalid ContentProperties specified.
  // E.g. expected ImageProperties, got BoundingBoxProperties.
  kMetadataInvalidContentPropertiesError,
  // Metadata is mandatory but was not found.
  // E.g. current task requires TFLite Model Metadata but none was found.
  kMetadataNotFoundError,
  // Associated TENSOR_AXIS_LABELS or TENSOR_VALUE_LABELS file is mandatory but
  // none was found or it was empty.
  // E.g. current task requires labels but none were found.
  kMetadataMissingLabelsError,
  // The ProcessingUnit for tokenizer is not correctly configured.
  // E.g BertTokenizer doesn't have a valid vocab file associated.
  kMetadataInvalidTokenizerError,

  // Input tensor(s) error codes.

  // Unexpected number of input tensors for the current task.
  // E.g. current task expects a single input tensor.
  kInvalidNumInputTensorsError = 300,
  // Unexpected input tensor dimensions for the current task.
  // E.g.: only 4D input tensors supported.
  kInvalidInputTensorDimensionsError,
  // Unexpected input tensor type for the current task.
  // E.g.: current task expects a uint8 pixel image as input.
  kInvalidInputTensorTypeError,
  // Unexpected input tensor bytes size.
  // E.g.: size in bytes does not correspond to the expected number of pixels.
  kInvalidInputTensorSizeError,
  // No correct input tensor found for the model.
  // E.g.: input tensor name is not part of the text model's input tensors.
  kInputTensorNotFoundError,

  // Output tensor(s) error codes.

  // Unexpected output tensor dimensions for the current task.
  // E.g.: only a batch size of 1 is supported.
  kInvalidOutputTensorDimensionsError = 400,
  // Unexpected input tensor type for the current task.
  // E.g.: multi-head model with different output tensor types.
  kInvalidOutputTensorTypeError,
  // No correct output tensor found for the model.
  // E.g.: output tensor name is not part of the text model's output tensors.
  kOutputTensorNotFoundError,
  // Unexpected number of output tensors for the current task.
  // E.g.: current task expects a single output tensor.
  kInvalidNumOutputTensorsError,

  // Image processing error codes.

  // Unspecified image processing failures.
  kImageProcessingError = 500,
  // Unexpected input or output buffer metadata.
  // E.g.: rotate RGBA buffer to Grayscale buffer by 90 degrees.
  kImageProcessingInvalidArgumentError,
  // Image processing operation failures.
  // E.g. libyuv rotation failed for an unknown reason.
  kImageProcessingBackendError,

  // Convenience error codes for condition checks during type casting.

  // Ensure it holds the least enum value.
  kErrorCodeFirst = kError,
  // Ensure it holds the greatest enum value.
  kErrorCodeLast = kImageProcessingBackendError,

};

// A `TfLiteSupportError` encapsulates an error code and a descriptive message to
// return in the event of an error being encountered in any TensorFlow Lite Task
// Library C API.
typedef struct TfLiteSupportError {
  // Holds the error code.
  enum TfLiteSupportErrorCode code;
  // Detailed description of the error.
  char *message;
} TfLiteSupportError;

void TfLiteErrorDelete(TfLiteSupportError *error);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_SUPPORT_C_COMMON_H_
