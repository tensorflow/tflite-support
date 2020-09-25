# RaggedTensors suppport in TFLite

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "ragged_tensor_to_tensor_tflite",
    srcs = ["ragged_tensor_to_tensor_tflite.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "@flatbuffers",
        "@org_tensorflow//tensorflow/core/util:ragged_to_dense_util_common",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/kernels/internal:types",
    ],
)

cc_test(
    name = "ragged_tensor_to_tensor_tflite_test",
    srcs = ["ragged_tensor_to_tensor_tflite_test.cc"],
    deps = [
        ":ragged_tensor_to_tensor_tflite",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)
