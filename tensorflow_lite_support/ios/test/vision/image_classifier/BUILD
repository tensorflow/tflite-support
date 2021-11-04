package(
    default_visibility = ["//tensorflow_lite_support:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

swift_library(
    name = "TFLImageClassifierTestLibrary",
    testonly = 1,
    srcs = ["TFLImageClassifierTests.swift"],
    data = [
        "//tensorflow_lite_support/cc/test/testdata/task/vision:test_images",
        "//tensorflow_lite_support/cc/test/testdata/task/vision:test_models",
    ],
    tags = TFL_DEFAULT_TAGS,
    deps = [
        "//tensorflow_lite_support/ios/task/vision:TFLImageClassifier",
        "//third_party/swift/xctest",
    ],
)

ios_unit_test(
    name = "TFLImageClassifierSwiftTest",
    minimum_os_version = TFL_MINIMUM_OS_VERSION,
    runner = tflite_ios_lab_runner("IOS_LATEST"),
    tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
    deps = [
        ":TFLImageClassifierTestLibrary",
    ],
)

# objc_library(
#     name = "TFLBertNLClassifierObjcTestLibrary",
#     testonly = 1,
#     srcs = ["Tests/TFLBertNLClassifierTest.m"],
#     data = [
#         "//tensorflow_lite_support/cc/test/testdata/task/text:bert_nl_classifier_models",
#     ],
#     tags = TFL_DEFAULT_TAGS,
#     deps = [
#         ":TFLBertNLClassifier",
#     ],
# )

# ios_unit_test(
#     name = "TFLBertNLClassifierObjcTest",
#     minimum_os_version = TFL_MINIMUM_OS_VERSION,
#     runner = tflite_ios_lab_runner("IOS_LATEST"),
#     tags = TFL_DEFAULT_TAGS + TFL_DISABLED_SANITIZER_TAGS,
#     deps = [
#         ":TFLBertNLClassifierObjcTestLibrary",
#     ],
# )