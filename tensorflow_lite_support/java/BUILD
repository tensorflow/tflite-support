# Description:
# TensorFlow Lite Support API in Java.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//third_party/tensorflow_lite_support/java:build_defs.bzl", "JAVACOPTS")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(b/156482505): The NOGPU target is a temporary target. Internally, people
# may already depend on "tensorflow-lite-support" so we shouldn't remove GPU
# from its dependency. We will have CLs to help users migrate. After migration
# is done, the "NOGPU" target will be removed.
android_library(
    name = "tensorflow-lite-support-nogpu",
    srcs = glob(["src/java/org/tensorflow/lite/support/**/*.java"]),
    javacopts = JAVACOPTS,
    manifest = "AndroidManifest.xml",
    deps = [
        "//third_party/java/checker_framework_annotations",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite",
    ],
)

# TODO(138904786): Split Java part and Android part to make the support library usable by pure Java.
# For new users: Please use "tensorflow-lite-support-nogpu" if possible, and
# additionally depends on "tensorflowlite_gpu" if needed.
android_library(
    name = "tensorflow-lite-support",
    srcs = glob(["src/java/org/tensorflow/lite/support/**/*.java"]),
    javacopts = JAVACOPTS,
    manifest = "AndroidManifest.xml",
    deps = [
        "//third_party/java/checker_framework_annotations",
        "//third_party/tensorflow/lite/java:tensorflowlite_gpu",  # unuseddeps: keep
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite",
    ],
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
        "//third_party/java/checker_framework_annotations",
    ],
)

android_library(
    name = "tensorflow-lite-support-precondition-lib-android",
    srcs = ["src/java/org/tensorflow/lite/support/common/SupportPreconditions.java"],
    javacopts = JAVACOPTS,
    manifest = "AndroidManifest.xml",
    deps = [
        "//third_party/java/checker_framework_annotations",
    ],
)
