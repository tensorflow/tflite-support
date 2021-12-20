package(
    default_visibility = ["//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLClassificationOptions",
    srcs = [
        "sources/TFLClassificationOptions.m",
    ],
    hdrs = [
        "sources/TFLClassificationOptions.h",
    ],
    module_name = "TFLClassificationOptions",
)

objc_library(
    name = "TFLCategory",
    srcs = [
        "sources/TFLCategory.m",
    ],
    hdrs = [
        "sources/TFLCategory.h",
    ],
    module_name = "TFLCategory",
)

objc_library(
    name = "TFLClassificationResult",
    srcs = [
        "sources/TFLClassificationResult.m",
    ],
    hdrs = [
        "sources/TFLClassificationResult.h",
    ],
    module_name = "TFLClassificationResult",
    deps = [
        ":TFLCategory",
    ],
)

objc_library(
    name = "TFLDetectionResult",
    srcs = [
        "sources/TFLDetectionResult.m",
    ],
    hdrs = [
        "sources/TFLDetectionResult.h",
    ],
    module_name = "TFLDetectionResult",
    deps = [
        ":TFLCategory",
    ],
)

objc_library(
    name = "TFLClassificationOptionsHelpers",
    srcs = [
        "sources/TFLClassificationOptions+Helpers.m",
    ],
    hdrs = [
        "sources/TFLClassificationOptions+Helpers.h",
    ],
    module_name = "TFLClassificationOptionsHelpers",
    deps = [
        "//tensorflow_lite_support/c/task/processor:classification_options",
        "//tensorflow_lite_support/ios:TFLCommon",
        "//tensorflow_lite_support/ios:TFLCommonUtils",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptions",
    ],
)
