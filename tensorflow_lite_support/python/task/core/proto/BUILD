# Placeholder for internal Python strict library compatibility macro.

package(
    default_visibility = [
        "//tensorflow_lite_support:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "configuration_pb2",
    srcs = ["configuration_pb2.py"],
    deps = ["@org_tensorflow//tensorflow/lite/experimental/acceleration/configuration:configuration_proto_external_py"],
)

py_library(
    name = "base_options_pb2",
    srcs = ["base_options_pb2.py"],
    deps = ["//tensorflow_lite_support/cc/task/core/proto:base_options_py_pb2"],
)
