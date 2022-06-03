/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/kernels/builtin_op_kernels.h"

namespace tflite {
namespace task {
// Create a  OpResolver, provides MutableOpResolver.
std::unique_ptr<OpResolver> CreateOpResolver() {  // NOLINT
    MutableOpResolver resolver;
    resolver.AddBuiltin(::tflite::BuiltinOperator_PACK, ::tflite::ops::builtin::Register_PACK());
    resolver.AddBuiltin(::tflite::BuiltinOperator_STRIDED_SLICE, ::tflite::ops::builtin::Register_STRIDED_SLICE());
    resolver.AddBuiltin(::tflite::BuiltinOperator_LOGISTIC, ::tflite::ops::builtin::Register_LOGISTIC());
    resolver.AddBuiltin(::tflite::BuiltinOperator_DIV, ::tflite::ops::builtin::Register_DIV());
    resolver.AddBuiltin(::tflite::BuiltinOperator_MAXIMUM, ::tflite::ops::builtin::Register_MAXIMUM());
    resolver.AddBuiltin(::tflite::BuiltinOperator_CAST, ::tflite::ops::builtin::Register_CAST());
    resolver.AddBuiltin(::tflite::BuiltinOperator_SUB, ::tflite::ops::builtin::Register_SUB());
    resolver.AddBuiltin(::tflite::BuiltinOperator_GATHER, ::tflite::ops::builtin::Register_GATHER());
    resolver.AddBuiltin(::tflite::BuiltinOperator_RESHAPE, ::tflite::ops::builtin::Register_RESHAPE());
    resolver.AddBuiltin(::tflite::BuiltinOperator_ADD, ::tflite::ops::builtin::Register_ADD());
    resolver.AddBuiltin(::tflite::BuiltinOperator_MUL, ::tflite::ops::builtin::Register_MUL());
    resolver.AddBuiltin(::tflite::BuiltinOperator_CONV_2D, ::tflite::ops::builtin::Register_CONV_2D());
    resolver.AddBuiltin(::tflite::BuiltinOperator_SHAPE, ::tflite::ops::builtin::Register_SHAPE());
    resolver.AddBuiltin(::tflite::BuiltinOperator_DEPTHWISE_CONV_2D, ::tflite::ops::builtin::Register_DEPTHWISE_CONV_2D());
    resolver.AddBuiltin(::tflite::BuiltinOperator_EXP, ::tflite::ops::builtin::Register_EXP());
    resolver.AddBuiltin(::tflite::BuiltinOperator_PAD, ::tflite::ops::builtin::Register_PAD());
    resolver.AddBuiltin(::tflite::BuiltinOperator_NON_MAX_SUPPRESSION_V4, ::tflite::ops::builtin::Register_NON_MAX_SUPPRESSION_V4());
    resolver.AddBuiltin(::tflite::BuiltinOperator_CONCATENATION, ::tflite::ops::builtin::Register_CONCATENATION());
    resolver.AddBuiltin(::tflite::BuiltinOperator_ARG_MAX, ::tflite::ops::builtin::Register_ARG_MAX());
    resolver.AddBuiltin(::tflite::BuiltinOperator_IF, ::tflite::ops::builtin::Register_IF());
    resolver.AddBuiltin(::tflite::BuiltinOperator_TILE, ::tflite::ops::builtin::Register_TILE());


    resolver.AddCustom(::tflite::"feawfa", ::tflite::ops::builtin::Register_TILE());

    // also need to register flex ops?
    // FlexScatterNd, FlexAddV2, FlexEqual, FlexStridedSlice

    return std::make_unique<MutableOpResolver>(resolver);
}

}  // namespace task
}  // namespace tflite
