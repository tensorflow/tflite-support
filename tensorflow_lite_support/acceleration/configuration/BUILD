package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "edgetpu_coral_plugin",
    srcs = ["edgetpu_coral_plugin.cc"],
    deps = [
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_glog//:glog",
        "@libedgetpu//tflite/public:edgetpu_c",
        "@libedgetpu//tflite/public:oss_edgetpu_direct_all",  # buildcleaner: keep
        "@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:delegate_registry",
    ],
    alwayslink = 1,  # For registration to always run.
)

cc_test(
    name = "edgetpu_coral_plugin_test",
    srcs = ["edgetpu_coral_plugin_test.cc"],
    data = [
        "//tensorflow_lite_support/acceleration/configuration/testdata:test_files",
    ],
    tags = [
        "manual",
        "notap",  # Requires edge TPU device.
    ],
    deps = [
        ":edgetpu_coral_plugin",
        "//tensorflow_lite_support/examples/task/vision/desktop/utils:image_utils",
        "@com_google_googletest//:gtest_main",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:configuration_fbs",
        "@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:delegate_registry",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
