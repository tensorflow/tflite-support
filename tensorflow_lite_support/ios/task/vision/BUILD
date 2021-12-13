package(
    default_visibility = ["//tensorflow_lite_support:__subpackages__"],
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

strip_c_api_include_path_prefix(
    name = "strip_c_api_include_path",
    hdr_labels = [
        "//tensorflow_lite_support/c/task/vision:object_detector.h",
        "//tensorflow_lite_support/c/task/processor:bounding_box.h",
        "//tensorflow_lite_support/c/task/vision/core:frame_buffer.h",
        "//tensorflow_lite_support/c/task/processor:detection_result.h",
        "//tensorflow_lite_support/c/task/processor:classification_options.h",
        "//tensorflow_lite_support/c/task/core:base_options.h",
        "//tensorflow_lite_support/c/task/processor:category.h",
        "//tensorflow_lite_support/c:common.h",
    ],
)

objc_library(
    name = "TFLImageClassifier",
    srcs = [
        "sources/TFLImageClassifier.m",
    ],
    hdrs = [
        "sources/TFLImageClassifier.h",
    ],
    module_name = "TFLImageClassifier",
    deps = [
        "//tensorflow_lite_support/c/task/vision:image_classifier",
        "//tensorflow_lite_support/ios:TFLCommonUtils",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptions",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptionsHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptions",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptionsHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationResult",
        "//tensorflow_lite_support/ios/task/processor/utils:TFLClassificationUtils",
        "//tensorflow_lite_support/ios/task/vision/utils:GMLImageUtils",
    ],
)

tflite_ios_framework(
    name = "TensorFlowLiteTaskObjectDetectionC_framework",
    hdrs = [
        ":base_options.h",
        ":bounding_box.h",
        ":classification_options.h",
        ":category.h", 
        ":detection_result.h",
        ":common.h",
        ":frame_buffer.h",
        ":object_detector.h",
    ],
    allowlist_symbols_file = "//tensorflow_lite_support/ios:allowlist_TensorFlowLiteTaskVision.txt",
    bundle_name = "TensorFlowLiteTaskObjectDetectionC",
    minimum_os_version = TFL_MINIMUM_OS_VERSION,
    deps = [
        "//tensorflow_lite_support/c/task/vision:object_detector",
    ],
)

objc_library(
    name = "TFLObjectDetector",
    hdrs = [
        "sources/TFLObjectDetector.h",
    ],
    module_name = "TFLObjectDetector",
    deps = [
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptions",
        "//tensorflow_lite_support/ios/task/processor:TFLClassificationOptions",
        "//tensorflow_lite_support/ios/task/processor:TFLDetectionResult",
        "//tensorflow_lite_support/odml/ios/image:MLImage",
    ],
)

