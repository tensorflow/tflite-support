package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tflite_engine",
    srcs = ["tflite_engine.cc"],
    hdrs = [
        "tflite_engine.h",
    ],
    deps = [
        ":external_file_handler",
        "//tensorflow_lite_support/cc:common",
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:tflite_wrapper",
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto_inc",
        "//tensorflow_lite_support/metadata/cc:metadata_extractor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/core/api",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/tools:verifier",
    ],
)

cc_library(
    name = "base_task_api",
    hdrs = [
        "base_task_api.h",
    ],
    deps = [
        ":tflite_engine",
        "//tensorflow_lite_support/cc:common",
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/port:tflite_wrapper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite/c:common",
    ],
)

cc_library(
    name = "task_api_factory",
    hdrs = [
        "task_api_factory.h",
    ],
    deps = [
        ":base_task_api",
        ":tflite_engine",
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto_inc",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/core/api",
        "@org_tensorflow//tensorflow/lite/kernels:op_macros",
    ],
)

cc_library(
    name = "task_utils",
    srcs = [
        "task_utils.cc",
    ],
    hdrs = [
        "task_utils.h",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite:type_to_tflitetype",
        "@org_tensorflow//tensorflow/lite/kernels:op_macros",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
    ],
)

cc_library(
    name = "category",
    hdrs = ["category.h"],
)

cc_library(
    name = "external_file_handler",
    srcs = ["external_file_handler.cc"],
    hdrs = ["external_file_handler.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_lite_support/cc:common",
        "//tensorflow_lite_support/cc/port:integral_types",
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto_inc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)
