package(
    default_visibility = ["//tensorflow_lite_support:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLClassificationOptions",
    hdrs = [
        "sources/TFLClassificationOptions.h",
    ],
    srcs = [
        "sources/TFLClassificationOptions.m",
    ],
    module_name = "TFLClassificationOptions",
)

objc_library(
    name = "TFLClassificationResult",
    hdrs = [
        "sources/TFLClassificationResult.h",
    ],
    srcs = [
        "sources/TFLClassificationResult.m",
    ],
    module_name = "TFLClassificationResult",
)

objc_library(
    name = "TFLClassificationOptionsHelpers",
    hdrs = [
        "sources/TFLClassificationOptions+Helpers.h",
    ],
    srcs = [
        "sources/TFLClassificationOptions+Helpers.m",
    ],
    deps = [
        "//tensorflow_lite_support/c/task/processor:classification_options",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptions",
        "//tensorflow_lite_support/ios:TFLCommon",
        "//tensorflow_lite_support/ios:TFLCommonUtils",
    ],
    module_name = "TFLClassificationOptionsHelpers",
)

