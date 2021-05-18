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

load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")


# Generates a statically linked framework for use on iOS.  To build fat, we need use the following
# command
#
# bazel build -c opt --config=ios_fat //tensorflow_lite_support/metadata/cc:TfliteMetadata_framework
#
ios_static_framework(
    name = "TfliteMetadata_framework",
    hdrs = [
        ":dl_tflite_metadata.h",
    ],
    bundle_name = "TfliteMetadata",
    minimum_os_version = "9.0",
    deps = [
        "//tensorflow_lite_support/metadata/cc:dl_tflite_metadata",
    ],
)

# This is a DL-Team custom library that just exposes metadata of interest for
# model inspection on the phone.
#
# We use this library rather than the metadata_extractor directly because the
# metadata_extractor.h header brings in a bunch of dependencies that
# would clutter the phone. Wrapping teh metadata_extractor also results in a
# pretty fat binary; so this is bare-bones (35K vs. 700K delta on the phone)!
#
cc_library(
  name="dl_tflite_metadata",
    srcs = [
         "dl_tflite_metadata.cc",
         "dl_tflite_metadata.h"
    ],
    deps = [
        "@flatbuffers",
        "//tensorflow_lite_support/metadata:metadata_schema_cc",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "metadata_extractor",
    srcs = ["metadata_extractor.cc"],
    hdrs = ["metadata_extractor.h"],
    deps = [
        "//tensorflow_lite_support/cc:common",
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/metadata:metadata_schema_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@flatbuffers",
        "@org_libzip//:zip",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
    ],
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

cc_library(
    name = "metadata_populator",
    srcs = ["metadata_populator.cc"],
    hdrs = ["metadata_populator.h"],
    deps = [
        "//tensorflow_lite_support/cc:common",
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/metadata:metadata_schema_cc",
        "//tensorflow_lite_support/metadata/cc/utils:zip_mem_file",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@flatbuffers//:runtime_cc",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@zlib//:zlib_minizip",
    ],
)
