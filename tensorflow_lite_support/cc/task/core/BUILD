package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tflite_engine",
    srcs = ["tflite_engine.cc"],
    hdrs = ["tflite_engine.h"],
    deps = [
        ":error_reporter",
        ":external_file_handler",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:kernel_api",
        "@org_tensorflow//tensorflow/lite/core/shims:common",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
        # The dependency on builtin_ops here is only for the default
        # value of the OpResolver parameter:
        #     std::unique_ptr<tflite::IterableOpResolver> resolver =
        #       absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>()
        # When linking statically, if the client of this library doesn't use
        # the default argument, this dependency does not cause all the builtin ops
        # to get included in the executable.
        "@org_tensorflow//tensorflow/lite/core/shims:builtin_ops",
        "@org_tensorflow//tensorflow/lite/core/shims:framework",
        "@org_tensorflow//tensorflow/lite/tools:verifier",
        "//tensorflow_lite_support/cc:common",
        "//tensorflow_lite_support/cc/port:configuration_proto_inc",
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:tflite_wrapper",
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto_inc",
        "//tensorflow_lite_support/metadata/cc:metadata_extractor",
    ],
)

cc_library(
    name = "base_task_api",
    hdrs = ["base_task_api.h"],
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
    hdrs = ["task_api_factory.h"],
    deps = [
        ":base_task_api",
        ":tflite_engine",
        "//tensorflow_lite_support/cc/port:configuration_proto_inc",
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto_inc",
        "@com_google_absl//absl/status",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/core/api:op_resolver",
        "@org_tensorflow//tensorflow/lite/kernels:op_macros",
    ],
)

cc_library(
    name = "task_utils",
    srcs = ["task_utils.cc"],
    hdrs = ["task_utils.h"],
    deps = [
        "//tensorflow_lite_support/metadata:metadata_schema_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
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

cc_library(
    name = "error_reporter",
    srcs = ["error_reporter.cc"],
    hdrs = ["error_reporter.h"],
    deps = [
        "@org_tensorflow//tensorflow/lite:minimal_logging",
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
    ],
)
