package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "bert_nl_classifier_options_proto",
    srcs = ["bert_nl_classifier_options.proto"],
    deps = [
        "//tensorflow_lite_support/cc/task/core/proto:base_options_proto",
    ],
)

cc_proto_library(
    name = "bert_nl_classifier_options_cc_proto",
    deps = [
        ":bert_nl_classifier_options_proto",
    ],
)

cc_library(
    name = "bert_nl_classifier_options_proto_inc",
    hdrs = ["bert_nl_classifier_options_proto_inc.h"],
    deps = [
        ":bert_nl_classifier_options_cc_proto",
        "//tensorflow_lite_support/cc/task/core/proto:base_options_proto_inc",
    ],
)
