package(
    default_visibility = ["//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

load(
    "@org_tensorflow//tensorflow/lite/ios:ios.bzl",
    "TFL_MINIMUM_OS_VERSION",
    "tflite_ios_framework",
)
load(
    "//tensorflow_lite_support/ios:ios.bzl",
    "strip_c_api_include_path_prefix",
)

# strip_c_api_include_path_prefix(
#     name = "strip_c_api_include_path",
#     hdr_labels = [ "//tensorflow_lite_support/ios/task/vision:TFLImageClassifierTest.h"],
# )

objc_library(
    name = "TFLImageClassifier",
    srcs = [
        "sources/TFLImageClassifier.m",
    ],
    hdrs = [
        "sources/TFLImageClassifier.h",
    ],
    features = ["-layering_check"],
    module_name = "TFLImageClassifier",
    deps = [
        "//tensorflow_lite_support/c/task/vision:image_classifier",
        "//tensorflow_lite_support/ios:TFLCommonUtils",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptions",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptionsHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptions",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptionsHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationResult",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationResultHelpers",
        "//tensorflow_lite_support/ios/task/vision/utils:GMLImageUtils",
        "//tensorflow_lite_support/odml/ios/image:MLImage",
    ],
)

objc_library(
    name = "TFLObjectDetector",
    srcs = [
        "sources/TFLObjectDetector.m",
    ],
    hdrs = [
        "sources/TFLObjectDetector.h",
    ],
    features = ["-layering_check"],
    module_name = "TFLObjectDetector",
    deps = [
        "//tensorflow_lite_support/c/task/vision:object_detector",
        "//tensorflow_lite_support/ios:TFLCommonUtils",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptions",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptionsHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptions",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptionsHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLDetectionResult",
        "//tensorflow_lite_support/ios/task/processor:TFLDetectionResultHelpers",
        "//tensorflow_lite_support/ios/task/vision/utils:GMLImageUtils",
        "//tensorflow_lite_support/odml/ios/image:MLImage",
    ],
)

objc_library(
    name = "TFLImageSegmenter",
    srcs = [
        "sources/TFLImageSegmenter.m",
        
    ],
    hdrs = [
        "sources/TFLImageSegmenter.h",
    ],
    features = ["-layering_check"],
    module_name = "TFLImageSegmenter",
    deps = [
        "//tensorflow_lite_support/c/task/vision:image_segmenter",
        "//tensorflow_lite_support/ios:TFLCommonUtils",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptions",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptionsHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLSegmentationResult",
        "//tensorflow_lite_support/ios/task/processor:TFLSegmentationResultHelpers",
        "//tensorflow_lite_support/ios/task/vision/utils:GMLImageUtils",
        "//tensorflow_lite_support/odml/ios/image:MLImage",
    ],
)

objc_library(
    name = "TFLImageClassifierTest",
    srcs = [
        "sources/TFLImageClassifierTest.mm",
    ],
    hdrs = [
        "sources/TFLImageClassifierTest.h",
    ],
    features = ["-layering_check"],
    module_name = "TFLImageClassifierTest",
    deps = [
        "//tensorflow_lite_support/cc/task/vision:image_classifier",
    ],
    copts = ["-ObjC++","-std=c++14"],
)

tflite_ios_framework(
    name = "TensorFlowLiteTaskImageClassifierTest_framework",
    hdrs = [
        "sources/TFLImageClassifierTest.h",
        # "sources/TFLObjectDetector.h",
        # "sources/TFLImageSegmenter.h",
        # "sources/TFLImageClassifier.h",
        # "sources/TFLImageClassifier.h",
        # "//tensorflow_lite_support/ios/task/processor/sources/TFLCategory.h"

    ],
    allowlist_symbols_file = ":allowlist_TensorFlowLiteTaskText.txt",
    bundle_name = "TensorFlowLiteTaskImageClassifierTest",
    minimum_os_version = TFL_MINIMUM_OS_VERSION,
    deps = [
        "//tensorflow_lite_support/ios/task/vision:TFLImageClassifierTest",
        # "//tensorflow_lite_support/ios/task/vision:TFLImageClassifier",
        # "//tensorflow_lite_support/ios/task/vision:TFLImageSegmenter",
        # "//tensorflow_lite_support/ios/task/vision:TFLObjectDetector",
    ],
)
