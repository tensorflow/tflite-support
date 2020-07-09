load("@org_tensorflow//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_portable_proto_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_proto_library(
    name = "external_file_proto",
    srcs = ["external_file.proto"],
    cc_api_version = 2,
)

tf_portable_proto_library(
    name = "external_file_portable_proto",
    config = "proto_config.pbtxt",
    header_outs = ["//tensorflow_lite_support/cc/task/core/proto/external_file.proto.h"],
    proto_deps = [
        ":external_file_proto",
    ],
)

cc_library(
    name = "external_file_proto_inc",
    hdrs = ["external_file_proto_inc.h"],
    deps = [":external_file_portable_proto"],
)
