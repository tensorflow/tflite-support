package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "external_file_proto",
    srcs = ["external_file.proto"],
)

cc_proto_library(
    name = "external_file_cc_proto",
    deps = [
        ":external_file_proto",
    ],
)

cc_library(
    name = "external_file_proto_inc",
    hdrs = ["external_file_proto_inc.h"],
    deps = [":external_file_cc_proto"],
)

proto_library(
    name = "base_options_proto",
    srcs = ["base_options.proto"],
    deps = [
        ":external_file_proto",
    ],
)

cc_proto_library(
    name = "base_options_cc_proto",
    deps = [
        ":base_options_proto",
    ],
)

cc_library(
    name = "base_options_proto_inc",
    hdrs = ["base_options_proto_inc.h"],
    deps = [
        ":base_options_cc_proto",
        ":external_file_proto_inc",
    ],
)
