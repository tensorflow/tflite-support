load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

tflite_jni_binary(
    name = "libbert_question_answerer_jni.so",
    srcs = [
        "bert_question_answerer_jni.cc",
    ],
    deps = [
        "//tensorflow_lite_support/cc/task/text/qa:bert_question_answerer",
        "//tensorflow_lite_support/cc/utils:jni_utils",
        "//third_party/java/jdk:jni",
    ],
)

cc_library(
    name = "bert_question_answerer_native",
    srcs = [
        ":libbert_question_answerer_jni.so",
    ],
)

android_library(
    name = "bert_question_answerer",
    srcs = glob(["*.java"]),
    deps = [
        ":bert_question_answerer_native",
        "//tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/task/core:base_task_api",
        "//third_party/java/android/android_sdk_linux/extras/android/compatibility/annotations",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite_java",
    ],
)
