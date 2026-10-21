load("//third_party/bazel_rules/rules_cc/cc:objc_library.bzl", "objc_library")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLStringUtil",
    srcs = [
        "sources/TFLStringUtil.mm",
    ],
    hdrs = [
        "sources/TFLStringUtil.h",
    ],
    module_name = "TFLStringUtil",
)

objc_library(
    name = "TFLCommonUtils",
    srcs = [
        "sources/TFLCommonUtils.m",
    ],
    hdrs = [
        "sources/TFLCommonUtils.h",
    ],
    module_name = "TFLCommonUtils",
    deps = [
        "//tensorflow_lite_support/ios:TFLCommon",
    ],
)
