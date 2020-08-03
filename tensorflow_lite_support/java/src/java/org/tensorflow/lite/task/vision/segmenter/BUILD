load("@build_bazel_rules_android//android:rules.bzl", "android_library")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["AndroidManifest.xml"])

android_library(
    name = "image_segmenter",
    srcs = glob(["*.java"]),
    manifest = "AndroidManifest.xml",
    deps = [
        "//tensorflow_lite_support/java:tensorflow-lite-support",
        "//tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core:base_task_api",
        "//tensorflow_lite_support/java/src/native/task/vision/segmenter:image_segmenter_native",
        "@com_google_auto_value",
        "@org_tensorflow//tensorflow/lite/java:tensorflowlite",
    ],
)
