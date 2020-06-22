load("@build_bazel_rules_android//android:rules.bzl", "android_library", "android_local_test")
load("//third_party/tensorflow/lite:build_def.bzl", "tflite_jni_binary")

package(
    default_testonly = 1,
)

licenses(["notice"])  # Apache License 2.0

android_library(
    name = "test_assets",
    assets = glob(["assets/**/*"]),
    assets_dir = "assets",
    manifest = "AndroidManifest.xml",
)

# wrap tflite_jni_binary within a cc_library, use the cc_library from
# android_library.
# native lib name is test_task_api_native_jni
tflite_jni_binary(
    name = "libtest_task_api_native_jni.so",
    testonly = 1,
    srcs = [
        "test_task_api_jni.cc",
    ],
    deps = [
        "//third_party/absl/status",
        "//third_party/java/jdk:jni",
    ],
)

cc_library(
    name = "test_task_api_native",
    srcs = [
        ":libtest_task_api_native_jni.so",
    ],
    alwayslink = 1,
)

android_library(
    name = "test_task_api",
    srcs = ["TestTaskApi.java"],
    deps = [
        ":test_task_api_native",
        "//third_party/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/task/core:base_task_api",
    ],
)

android_local_test(
    name = "BaseTaskApiTest",
    srcs = [
        "BaseTaskApiTest.java",
    ],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    test_class = "org.tensorflow.lite.support.task.core.BaseTaskApiTest",
    deps = [
        ":test_assets",
        ":test_task_api",
        "//third_party/java/junit:junit-android",
        "//third_party/java/mockito:mockito-android",
        "//third_party/java/truth:truth-android",
    ],
)
