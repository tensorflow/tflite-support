load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "statusor",
    hdrs = [
        "statusor.h",
    ],
    deps = [
        "//tensorflow_lite_support/cc/port/default:statusor",
    ],
)

cc_library(
    name = "status_macros",
    hdrs = [
        "status_macros.h",
    ],
    deps = [
        "//tensorflow_lite_support/cc/port/default:status_macros",
    ],
)

cc_library(
    name = "configuration_proto_inc",
    hdrs = ["configuration_proto_inc.h"],
    deps = ["@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:configuration_cc_proto"],
)

cc_library_with_tflite(
    name = "tflite_wrapper",
    hdrs = ["tflite_wrapper.h"],
    deps = ["//tensorflow_lite_support/cc/port/default:tflite_wrapper"],
)

cc_library(
    name = "integral_types",
    hdrs = ["integral_types.h"],
)

cc_library(
    name = "gtest",
    testonly = 1,
    hdrs = [
        "gmock.h",
        "gtest.h",
    ],
    deps = [
        "//testing/base/public:gunit_for_library_testonly",
    ],
)

cc_library(
    name = "gtest_main",
    testonly = 1,
    hdrs = [
        "benchmark.h",
        "gmock.h",
        "gtest.h",
    ],
    deps = [
        "@com_google_googletest//:gtest_main",
    ],
)
