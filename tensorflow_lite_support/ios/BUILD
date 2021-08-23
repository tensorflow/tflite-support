# TensorFlow Lite Task Library - Text

load(
    "@org_tensorflow//tensorflow/lite/ios:ios.bzl",
    "TFL_MINIMUM_OS_VERSION",
    "tflite_ios_framework",
)
load(
    "//tensorflow_lite_support/ios:ios.bzl",
    "strip_c_api_include_path_prefix",
)

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

strip_c_api_include_path_prefix(
    name = "strip_c_api_include_path",
    hdr_labels = [
        "//tensorflow_lite_support/c/task/text:bert_nl_classifier.h",
        "//tensorflow_lite_support/c/task/text:nl_classifier.h",
        "//tensorflow_lite_support/c/task/text:nl_classifier_common.h",
        "//tensorflow_lite_support/c/task/text:bert_question_answerer.h",
    ],
)

# This target builds a monolithic static framework for the TFLite Text API,
# which includes the TFLite runtime in it.
#
# bazel build -c opt --config=ios_fat //tensorflow_lite_support/ios:TensorFlowLiteTaskTextC_framework
tflite_ios_framework(
    name = "TensorFlowLiteTaskTextC_framework",
    hdrs = [
        ":bert_nl_classifier.h",
        ":bert_question_answerer.h",
        ":nl_classifier.h",
        ":nl_classifier_common.h",
    ],
    allowlist_symbols_file = ":allowlist_TensorFlowLiteTaskText.txt",
    bundle_name = "TensorFlowLiteTaskTextC",
    minimum_os_version = TFL_MINIMUM_OS_VERSION,
    deps = [
        "//tensorflow_lite_support/c/task/text:bert_nl_classifier",
        "//tensorflow_lite_support/c/task/text:bert_question_answerer",
        "//tensorflow_lite_support/c/task/text:nl_classifier",
    ],
)
