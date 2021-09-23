package(
    default_visibility = ["//tensorflow_lite_support:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "common_error",
    srcs = ["common_error.cc"],
    hdrs = ["common_error.h"],
)

cc_library(
    name = "common_error_utils",
    srcs = ["common_error_utils.cc"],
    hdrs = ["common_error_utils.h"],
    deps = [
        ":common_error",
        "//tensorflow_lite_support/cc:common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ]
)
