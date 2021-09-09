load(
    "@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl",
    "cc_library_with_tflite",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library_with_tflite(
    name = "audio_classifier",
    srcs = ["audio_classifier.cc"],
    hdrs = ["audio_classifier.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite/core/shims:builtin_ops",
        "//tensorflow_lite_support/cc/task/processor:audio_preprocessor",
        "//tensorflow_lite_support/cc/task/core:base_task_api",
        "//tensorflow_lite_support/cc/task/core:task_api_factory",
    ],
    deps = [
        "//tensorflow_lite_support/cc:common",
        "//tensorflow_lite_support/cc/port:integral_types",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/audio/core:audio_buffer",
        "//tensorflow_lite_support/cc/task/audio/proto:audio_classifier_options_cc_proto",
        "//tensorflow_lite_support/cc/task/audio/proto:class_proto_inc",
        "//tensorflow_lite_support/cc/task/audio/proto:classifications_proto_inc",
        "//tensorflow_lite_support/cc/task/core:classification_head",
        "//tensorflow_lite_support/cc/task/core:label_map_item",
        "//tensorflow_lite_support/cc/task/core:task_utils",
        "//tensorflow_lite_support/metadata:metadata_schema_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/lite/c:c_api_types",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
    ],
    alwayslink = 1,
)
