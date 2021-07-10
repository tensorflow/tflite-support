load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

# TODO(b/193127200): Use tensorflow_lite_support:internal (TBA) instead.
package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library_with_tflite(
    name = "processor",
    srcs = ["processor.cc"],
    hdrs = ["processor.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite/core/shims:common",
        "//tensorflow_lite_support/cc/task/core:tflite_engine",
    ],
    deps = [
        "//tensorflow_lite_support/cc:common",
        "//tensorflow_lite_support/cc/port:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)
