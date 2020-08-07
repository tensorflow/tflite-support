# Placeholder for internal Python strict compatibility macro.

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "whitespace_tokenizer",
    srcs = ["whitespace_tokenizer.cc"],
    hdrs = ["whitespace_tokenizer.h"],
    deps = [
        "//third_party/utf",
        "@org_tensorflow//tensorflow/lite:context",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
    ],
)

cc_library(
    name = "whitespace_tokenizer_op_resolver",
    srcs = ["whitespace_tokenizer_op_resolver.cc"],
    hdrs = ["whitespace_tokenizer_op_resolver.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":whitespace_tokenizer",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_test(
    name = "whitespace_tokenizer_test",
    srcs = ["whitespace_tokenizer_test.cc"],
    deps = [
        ":whitespace_tokenizer",
        "//testing/base/public:gunit_main",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

py_test(
    name = "whitespace_tokenizer_py_test",
    srcs = ["whitespace_tokenizer_test.py"],
    data = [
        "testdata/whitespace_tokenizer_flex_delegate.tflite",
        "testdata/whitespace_tokenizer_to_ragged_1d_input.tflite",
        "testdata/whitespace_tokenizer_to_ragged_2d_input.tflite",
        "testdata/whitespace_tokenizer_to_tensor.tflite",
    ],
    main = "whitespace_tokenizer_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":whitespace_tokenizer_op_resolver",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_text",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow/lite/delegates/flex:delegate",
        "@org_tensorflow//tensorflow/lite/python:interpreter",
        "@org_tensorflow//tensorflow/python:platform",
    ],
)

cc_library(
    name = "ngrams",
    srcs = ["ngrams.cc"],
    hdrs = ["ngrams.h"],
    deps = [
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:context",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
    ],
)

cc_library(
    name = "ngrams_op_resolver",
    srcs = ["ngrams_op_resolver.cc"],
    hdrs = ["ngrams_op_resolver.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ngrams",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
    alwayslink = 1,
)

cc_test(
    name = "ngrams_test",
    srcs = ["ngrams_test.cc"],
    deps = [
        ":ngrams",
        "//testing/base/public:gunit_main",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

py_test(
    name = "ngrams_py_test",
    srcs = ["ngrams_test.py"],
    main = "ngrams_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ngrams_op_resolver",
        "//tensorflow_lite_support/custom_ops/python:tflite_text_api",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_text",
        "@absl_py//absl/logging",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow/lite/python:interpreter",
        "@org_tensorflow//tensorflow/python:platform",
    ],
)
