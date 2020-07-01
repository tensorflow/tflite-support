# Description:
# TensorFlow Lite Support API in Java.

load("@org_tensorflow//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("@build_bazel_rules_android//android:rules.bzl", "android_library")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

android_library(
    name = "tensorflow-lite-support",
    srcs = glob(
        ["src/java/org/tensorflow/lite/support/**/*.java"],
        exclude = ["src/java/org/tensorflow/lite/support/task/**/*.java"],
    ),
    javacopts = JAVACOPTS,
    deps = [
        "@org_checkerframework_qual",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite",
    ],
)

# TODO(b/156482505): Remove this target.
alias(
    name = "tensorflow-lite-support-nogpu",
    actual = ":tensorflow-lite-support",
)

# This alias matches the style of lite/java naming for android_library targets. We keep the
# `tensorflow-lite-support` variant to match the associated .aar library name output style.
alias(
    name = "tensorflowlite_support",
    actual = ":tensorflow-lite-support",
)

java_library(
    name = "tensorflow-lite-support-precondition",
    srcs = ["src/java/org/tensorflow/lite/support/common/SupportPreconditions.java"],
    javacopts = JAVACOPTS,
    deps = [
        "@org_checkerframework_qual",
    ],
)

android_library(
    name = "tensorflow-lite-support-precondition-lib-android",
    srcs = ["src/java/org/tensorflow/lite/support/common/SupportPreconditions.java"],
    javacopts = JAVACOPTS,
    manifest = "AndroidManifest.xml",
    deps = [
        "@org_checkerframework_qual",
    ],
)
