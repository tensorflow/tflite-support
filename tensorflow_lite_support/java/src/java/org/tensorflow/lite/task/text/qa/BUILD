load("@org_tensorflow//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@org_tensorflow//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

filegroup(
    name = "bert_question_answerer_src",
    srcs = glob(["**/*.java"]),
)

# Default target that uses BuiltInOpResolver, registers all built-in OPs.
android_library(
    name = "bert_question_answerer",
    exports = [
        ":bert_question_answerer_java",
        "//tensorflow_lite_support/java/src/native/task/text/qa:bert_question_answerer_native",
    ],
)

# Java-only target, need to be used together with a native target similar to
# //third_party/tensorflow_lite_support/java/src/native/task/text/qa:bert_question_answerer_native.
# Use this target when you want to provide a MutableOpResolver with customized
# OPs and/or a subset of BuiltInOps to reduce binary size.
android_library(
    name = "bert_question_answerer_java",
    srcs = glob(["*.java"]),
    javacopts = JAVACOPTS,
    deps = [
        "//tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core:base_task_api",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite_java",
    ],
)

# AAR target for OSS release.
#
# bazel build -c opt --config=monolithic --config=android_arm64 --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
#   tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/text/qa:bert-question-answerer
aar_with_jni(
    name = "bert-question-answerer",
    android_library = ":bert_question_answerer",
)
