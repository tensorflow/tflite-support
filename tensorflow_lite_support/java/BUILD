# Description:
# TensorFlow Lite Support API in Java.

load("@org_tensorflow//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("@org_tensorflow//tensorflow/lite/core/shims:repackage_rule.bzl", "repackage_rule")
load("@build_bazel_rules_android//android:rules.bzl", "android_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files([
    "AndroidManifest.xml",
    "default_version_script.lds",
    "debug_version_script.lds",
])

filegroup(
    name = "java_srcs",
    srcs = glob(["src/java/org/tensorflow/lite/support/**/*.java"]),
)

# Android Library target for TFLite Support Library. It depends on TensorFlow
# Lite runtime (tensorflow/lite/java:tensorflowlite). If you don't want to
# introduce the native library into dependencies, use
# "tensorflowlite_support_java" instead, which depends on
# tensorflow/lite/java:tensorflowlite_java.
android_library(
    name = "tensorflowlite_support",
    srcs = glob(
        ["src/java/org/tensorflow/lite/support/**/*.java"],
    ),
    javacopts = ["-source 7 -target 7"],
    manifest = "AndroidManifest.xml",
    deps = [
        "@com_google_auto_value",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_android_odml_image",
        "@org_checkerframework_qual",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite",
    ],
)

android_library(
    name = "tensorflowlite_support_java",
    srcs = glob(["src/java/org/tensorflow/lite/support/**/*.java"]),
    javacopts = ["-source 7 -target 7"],
    manifest = "AndroidManifest.xml",
    # LINT.IfChange(dep)
    deps = [
        "@com_google_auto_value",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_android_odml_image",
        "@org_checkerframework_qual",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite_java",
    ],
    # LINT.ThenChange(<INTERNAL>/release/build_support_pom.sh:dep)
)

# This alias matches the associated .aar library name output style.
alias(
    name = "tensorflow-lite-support",
    actual = ":tensorflowlite_support",
)

java_library(
    name = "tensorflowlite_support_precondition_lib",
    srcs = ["src/java/org/tensorflow/lite/support/common/SupportPreconditions.java"],
    javacopts = JAVACOPTS,
    deps = [
        "@org_checkerframework_qual",
    ],
)

android_library(
    name = "tensorflowlite_support_precondition",
    srcs = ["src/java/org/tensorflow/lite/support/common/SupportPreconditions.java"],
    javacopts = ["-source 7 -target 7"],
    manifest = "AndroidManifest.xml",
    deps = [
        "@org_checkerframework_qual",
    ],
)

#-------------------------------------------------------------------------------
# Targets that repackage the sources in a new android_library that can be
# used on top of the shims.

repackage_rule(
    name = "java_srcs_gms_client",
    srcs = ["//tensorflow_lite_support/java:java_srcs"],
    former_directory = "org.tensorflow.lite.support",
    new_directory = "com.google.android.gms.tflite.support",
    rewrites = {
        "org\\.tensorflow\\.lite": "com.google.android.gms.tflite",
    },
)

android_library(
    name = "tensorflowlite_support_gms_client",
    srcs = [":java_srcs_gms_client"],
    javacopts = ["-source 7 -target 7"],
    deps = [
        "@com_google_auto_value",
        "@maven//:androidx_annotation_annotation",
        "@maven//:com_google_android_odml_image",
        "@org_checkerframework_qual",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite_java_gms_client",
    ],
)
