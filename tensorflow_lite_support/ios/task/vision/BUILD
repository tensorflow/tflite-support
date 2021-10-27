package(
    default_visibility = ["//tensorflow_lite_support:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLImageClassifier",
    hdrs = [
        "apis/TFLImageClassifier.h",
    ],
    deps = [
        "//tensorflow_lite_support/odml/ios/image:MLImage",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptions",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptions",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationResult"
    ],
    module_name = "TFLImageClassifier",
)
