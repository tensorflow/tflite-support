load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "image_classifier_native",
    srcs = [
        ":libimage_classifier_jni.so",
    ],
)

tflite_jni_binary(
    name = "libimage_classifier_jni.so",
    srcs = [
        "image_classifier_jni.cc",
    ],
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/vision:image_classifier",
        "//tensorflow_lite_support/cc/task/vision/core:frame_buffer",
        "//tensorflow_lite_support/cc/task/vision/proto:classifications_proto_inc",
        "//tensorflow_lite_support/cc/task/vision/proto:image_classifier_options_proto_inc",
        "//tensorflow_lite_support/cc/task/vision/utils:frame_buffer_common_utils",
        "//tensorflow_lite_support/cc/utils:jni_utils",
        "//third_party/java/jdk:jni",
    ],
)
