package(
    default_visibility = ["//tensorflow_lite_support:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLBaseOptions",
    hdrs = [
        "sources/TFLBaseOptions.h",
    ],
    srcs = [
        "sources/TFLBaseOptions.m",
    ],
    module_name = "TFLBaseOptions",
)

objc_library(
    name = "TFLBaseOptionsHelpers",
    hdrs = [
        "sources/TFLBaseOptions+Helpers.h",
    ],
    srcs = [
        "sources/TFLBaseOptions+Helpers.m",
    ],
    deps = [
        "//tensorflow_lite_support/c/task/core:base_options",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptions",
    ],
    module_name = "TFLBaseOptionsHelpers",
)

