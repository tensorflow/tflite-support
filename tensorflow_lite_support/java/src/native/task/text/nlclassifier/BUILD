load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "nl_classifier_jni.cc",
])

# Default native target for nl_classifier to provide BuiltInOpResolver.
cc_library(
    name = "nl_classifier_native",
    srcs = [
        ":libtask_text_jni.so",
    ],
)

# Note: "libtask_text_jni" is hardcoded in Java to look up the .so, therefore
# the name should remain the same when creating customized version of
# nl_classifier_native
tflite_jni_binary(
    name = "libtask_text_jni.so",
    linkscript = "//tensorflow_lite_support/java:default_version_script.lds",
    deps = [
        ":native_without_resolver",
        "//tensorflow_lite_support/java/src/native/task/core:builtin_op_resolver",
    ],
)

# Shared native logic for NLClassifier. Combine this target and customized
# version of op_resolver to build customized nl_classifier_native target.
cc_library(
    name = "native_without_resolver",
    srcs = [
        "nl_classifier_jni.cc",
    ],
    deps = [
        ":nl_classifier_jni_utils",
        "//tensorflow_lite_support/cc/task/text/nlclassifier:nl_classifier",
        "//tensorflow_lite_support/cc/utils:jni_utils",
        "//tensorflow_lite_support/java/jni",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "nl_classifier_jni_utils",
    srcs = [
        "nl_classifier_jni_utils.cc",
    ],
    hdrs = [
        "nl_classifier_jni_utils.h",
    ],
    deps = [
        "//tensorflow_lite_support/cc/task/text/nlclassifier:nl_classifier",
        "//tensorflow_lite_support/cc/utils:jni_utils",
        "//tensorflow_lite_support/java/jni",
    ],
)
