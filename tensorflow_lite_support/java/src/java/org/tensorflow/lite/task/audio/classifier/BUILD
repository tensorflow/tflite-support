load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@org_tensorflow//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("@org_tensorflow//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

# Default target that uses BuiltInOpResolver, registers all built-in OPs.
android_library(
    name = "audio_classifier",
    exports = [
        ":audio_classifier_java",
        "//tensorflow_lite_support/java/src/native/task/audio/classifier:audio_classifier_native",
    ],
)

# Java-only target, needs to be used together with a native target similar to
# //third_party/tensorflow_lite_support/java/src/native/task/audio/classifier:audio_classifier_native.
# Use this target when you want to provide a MutableOpResolver with customized
# OPs and/or a subset of BuiltInOps to reduce binary size.
android_library(
    name = "audio_classifier_java",
    srcs = ["AudioClassifier.java"],
    javacopts = JAVACOPTS,
    deps = [
        "//tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core:base_task_api",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite_java",
    ],
)

# AAR target for OSS release.
#
# bazel build -c opt --config=monolithic --config=android_arm64 --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
#   tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/audio/classifier:audio-classifier
aar_with_jni(
    name = "audio-classifier",
    android_library = ":audio_classifier",
)
