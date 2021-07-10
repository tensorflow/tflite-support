package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "gpu_plugin",
    deps = [
        "@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:gpu_plugin",
    ],
    alwayslink = 1,  # For registration to always run.
)

cc_library(
    name = "nnapi_plugin",
    deps = [
        "@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:nnapi_plugin",
    ],
    alwayslink = 1,  # For registration to always run.
)

cc_library(
    name = "hexagon_plugin",
    deps = [
        "@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:hexagon_plugin",
    ],
    alwayslink = 1,  # For registration to always run.
)

cc_library(
    name = "xnnpack_plugin",
    deps = [
        "@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:xnnpack_plugin",
    ],
    alwayslink = 1,  # For registration to always run.
)

# To use the Coral_edgetpu_plugin externally, add the following flags to the bazel command.
# --linkopt=-lusb-1.0  --define darwinn_portable=1.
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

# To test it externally, plugin a Coral device, and run the following command:
# bazel test tensorflow_lite_support/acceleration/configuration:edgetpu_coral_plugin_test \
# --define darwinn_portable=1 --linkopt=-lusb-1.0
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
