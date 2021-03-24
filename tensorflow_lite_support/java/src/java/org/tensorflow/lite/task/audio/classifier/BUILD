load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@org_tensorflow//tensorflow/java:build_defs.bzl", "JAVACOPTS")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

android_library(
    name = "audio_classifier_java",
    srcs = glob(["*.java"]),
    javacopts = JAVACOPTS,
    deps = [
        "//tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core:base_task_api",
    ],
)
