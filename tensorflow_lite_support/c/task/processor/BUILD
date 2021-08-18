load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "classification_result_c_api",
    srcs = [
        "classification_result_c_api.cc",
    ],
    hdrs = ["classification_result_c_api.h"],
)