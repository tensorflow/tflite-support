load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

package(
    default_testonly = 1,
    default_visibility = ["//tensorflow_lite_support:users"],
)

licenses(["notice"])  # Apache License 2.0

cc_library(
    name = "custom_op_resolver",
    srcs = ["custom_op_resolver.cc"],
    deps = [
        "//tensorflow_lite_support/cc/test/task/text/nlclassifier:nl_classifier_test_utils",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

tflite_jni_binary(
    name = "libnl_classifier_jni.so",
    testonly = 1,
    deps = [
        ":custom_op_resolver",
        "//tensorflow_lite_support/java/src/native/task/text/nlclassifier:native_without_resolver",
    ],
)

cc_library(
    name = "nl_classifier_custom_ops_native",
    srcs = [
        ":libnl_classifier_jni.so",
    ],
)
