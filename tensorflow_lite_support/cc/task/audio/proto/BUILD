load("//tools/build_defs/js/jspb_proto_library:rules.bzl", "jspb_proto_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# AudioClassifier protos.

proto_library(
    name = "audio_classifier_options_proto",
    srcs = ["audio_classifier_options.proto"],
    deps = [
        "//tensorflow_lite_support/cc/task/core/proto:base_options_proto",
    ],
)

jspb_proto_library(
    name = "audio_classifier_options_jspb_proto",
    visibility = [
        "//tensorflow_lite_support/examples/task/audio/web/audio_classifier:__pkg__",
        "//tensorflow_lite_support/web/task/audio:__pkg__",
    ],
    deps = [":audio_classifier_options_proto"],
)

cc_proto_library(
    name = "audio_classifier_options_cc_proto",
    deps = [
        ":audio_classifier_options_proto",
    ],
)

cc_library(
    name = "classifications_proto_inc",
    hdrs = ["classifications_proto_inc.h"],
    deps = [
        "//tensorflow_lite_support/cc/task/core/proto:classifications_cc_proto",
    ],
)

cc_library(
    name = "class_proto_inc",
    hdrs = ["class_proto_inc.h"],
    deps = [
        "//tensorflow_lite_support/cc/task/core/proto:class_cc_proto",
    ],
)

proto_library(
    name = "audio_embedder_options_proto",
    srcs = ["audio_embedder_options.proto"],
    deps = [
        "//tensorflow_lite_support/cc/task/core/proto:base_options_proto",
        "//tensorflow_lite_support/cc/task/processor/proto:embedding_options_proto",
    ],
)

cc_proto_library(
    name = "audio_embedder_options_cc_proto",
    deps = [":audio_embedder_options_proto"],
)
