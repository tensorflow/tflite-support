load("//tensorflow_lite_support/metadata:build_defs.bzl", "stamp_metadata_parser_version")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

stamp_metadata_parser_version(
    name = "metadata_parser_h",
    srcs = ["metadata_parser.h.template"],
    outs = ["metadata_parser.h"],
)

cc_library(
    name = "metadata_version",
    srcs = ["metadata_version.cc"],
    hdrs = [
        "metadata_version.h",
        ":metadata_parser_h",
    ],
    deps = [
        "//tensorflow_lite_support/metadata:metadata_schema_cc",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels/internal:compatibility",
        "@org_tensorflow//tensorflow/lite/tools:logging",
    ],
)
