package(
    default_visibility = ["//tensorflow_lite_support:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLImageClassifier",
    hdrs = [
        "sources/TFLImageClassifier.h",
    ],
    srcs = [
        "sources/TFLImageClassifier.m",
    ],
    module_name = "TFLImageClassifier",
    deps = [
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptions",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptions",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationResult",
        "//tensorflow_lite_support/ios/task/processor/utils:TFLClassificationUtils",
        "//tensorflow_lite_support/ios:TFLCommonUtils",
        "//tensorflow_lite_support/ios/task/vision/utils:GMLImageUtils",
        "//tensorflow_lite_support/c/task/vision:image_classifier",
    ],
)
