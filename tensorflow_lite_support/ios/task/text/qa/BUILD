load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load("//third_party/tensorflow/lite/experimental/ios:ios.bzl", "TFL_DEFAULT_TAGS", "TFL_DISABLED_SANITIZER_TAGS", "TFL_MINIMUM_OS_VERSION")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_unit_test")
load("//third_party/tensorflow/lite:special_rules.bzl", "tflite_ios_lab_runner")

package(
    default_visibility = ["//third_party/tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLBertQuestionAnswerer",
    srcs = ["Sources/TFLBertQuestionAnswerer.mm"],
    hdrs = ["Sources/TFLBertQuestionAnswerer.h"],
    module_name = "TFLBertQuestionAnswerer",
    deps = [
        "//third_party/objective_c/google_toolbox_for_mac:GTM_Defines",
        "//third_party/tensorflow_lite_support/cc/task/text/qa:bert_question_answerer",
        "//third_party/tensorflow_lite_support/ios/utils:TFLStringUtil",
    ],
)

swift_library(
    name = "TFLBertQuestionAnswererTestLibrary",
    testonly = 1,
    srcs = glob(["Tests/*.swift"]),
    data = [
        "//third_party/tensorflow_lite_support/cc/testdata:albert_model",
        "//third_party/tensorflow_lite_support/cc/testdata:mobile_bert_model",
    ],
    tags = TFL_DEFAULT_TAGS,
    deps = [
        ":TFLBertQuestionAnswerer",
        "//third_party/swift/xctest",
    ],
)

ios_unit_test(
    name = "TFLBertQuestionAnswererTest",
    size = "large",
    minimum_os_version = TFL_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":TFLBertQuestionAnswererTestLibrary",
    ],
)
