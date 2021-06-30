load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Common vision protos.

proto_library(
    name = "classification_proto",
    srcs = ["classification.proto"],
)

cc_proto_library(
    name = "classification_cc_proto",
    deps = [
        ":classification_proto",
    ],
)

proto_library(
    name = "embedding_proto",
    srcs = ["embedding.proto"],
)

cc_proto_library(
    name = "embedding_cc_proto",
    deps = [
        ":embedding_proto",
    ],
)
