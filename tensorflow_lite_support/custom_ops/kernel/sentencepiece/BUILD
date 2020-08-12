# Memorymappable, WASM compilable, implementation of the encoder.
#

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load(
    "//tensorflow_lite_support/custom_ops/kernel/sentencepiece:native.bzl",
    "micore_tf_copts",
    "micore_tf_deps",
)
load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//tensorflow_lite_support:users",
    ],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "testdata",
    srcs = glob([
        "testdata/**",
    ]),
)

flatbuffer_cc_library(
    name = "encoder_config",
    srcs = [
        "encoder_config.fbs",
    ],
)

cc_library(
    name = "utils",
    srcs = [
    ],
    hdrs = [
        "utils.h",
    ],
)

cc_library(
    name = "double_array_trie",
    srcs = [
    ],
    hdrs = [
        "double_array_trie.h",
    ],
    deps = [
        ":encoder_config",
        ":utils",
    ],
)

cc_library(
    name = "double_array_trie_builder",
    srcs = [
        "double_array_trie_builder.cc",
    ],
    hdrs = [
        "double_array_trie_builder.h",
    ],
    deps = [
        ":encoder_config",
        "//third_party/darts_clone",
    ],
)

cc_test(
    name = "double_array_trie_test",
    srcs = [
        "double_array_trie_test.cc",
    ],
    deps = [
        ":double_array_trie",
        ":double_array_trie_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_converter",
    srcs = [
        "model_converter.cc",
    ],
    hdrs = [
        "model_converter.h",
    ],
    deps = [
        ":double_array_trie_builder",
        ":encoder_config",
        "//third_party/sentencepiece/src:sentencepiece_model_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "optimized_encoder",
    srcs = [
        "optimized_encoder.cc",
    ],
    hdrs = [
        "optimized_encoder.h",
    ],
    deps = [
        ":double_array_trie",
        ":encoder_config",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer_h",
    hdrs = [
        "sentencepiece_tokenizer.h",
    ],
)

cc_library(
    name = "sentencepiece_tokenizer_op",
    srcs = ["sentencepiece_tokenizer_op.cc"],
    copts = micore_tf_copts(),
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":sentencepiece_tokenizer_h",
        ":optimized_encoder",
    ] + micore_tf_deps(),
    alwayslink = 1,
)

cc_library(
    name = "sentencepiece_tokenizer_tflite",
    srcs = ["sentencepiece_tokenizer_tflite.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps =
        [
            ":optimized_encoder",
            ":sentencepiece_tokenizer_h",
            "@flatbuffers",
            "@org_tensorflow//tensorflow/lite:framework",
            "@org_tensorflow//tensorflow/lite:string_util",
            "@org_tensorflow//tensorflow/lite/c:common",
            "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
            "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        ],
)

tf_gen_op_wrapper_py(
    name = "gen_sentencepiece_tokenizer_op",
    out = "gen_sentencepiece_tokenizer_op.py",
    op_whitelist = ["TFSentencepieceTokenizeOp"],
    deps = [":sentencepiece_tokenizer_op"],
)

cc_test(
    name = "optimized_encoder_test",
    srcs = [
        "optimized_encoder_test.cc",
    ],
    data = [
        ":testdata",
    ],
    deps = [
        ":double_array_trie_builder",
        ":encoder_config",
        ":model_converter",
        ":optimized_encoder",
        "//third_party/sentencepiece/src:sentencepiece_cc_proto",
        "//third_party/sentencepiece/src:sentencepiece_processor",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

cc_library(
    name = "py_tflite_registerer",
    srcs = ["py_tflite_registerer.cc"],
    hdrs = ["py_tflite_registerer.h"],
    deps = [
        ":sentencepiece_tokenizer_tflite",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
    alwayslink = 1,
)

pybind_extension(
    name = "pywrap_tflite_registerer",
    srcs = [
        "pywrap_tflite_registerer.cc",
    ],
    hdrs = ["py_tflite_registerer.h"],
    additional_exported_symbols = ["TFLite_SentencepieceTokenizerRegisterer"],
    module_name = "pywrap_tflite_registerer",
    deps = [
        ":py_tflite_registerer",
        "//third_party/pybind11",
        "//third_party/python_runtime:headers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
