load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

# TODO(b/193127200): Use tensorflow_lite_support:internal (TBA) instead.
package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library_with_tflite(
    name = "processor",
    srcs = ["processor.cc"],
    hdrs = ["processor.h"],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite/core/shims:common",
        "//tensorflow_lite_support/cc/task/core:tflite_engine",
    ],
    deps = [
        "//tensorflow_lite_support/cc:common",
        "//tensorflow_lite_support/cc/port:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library_with_tflite(
    name = "image_preprocessor",
    srcs = ["image_preprocessor.cc"],
    hdrs = ["image_preprocessor.h"],
    tflite_deps = [
        ":processor",
        "//tensorflow_lite_support/cc/task/vision/utils:image_tensor_specs",
    ],
    deps = [
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/task/core:task_utils",
        "//tensorflow_lite_support/cc/task/vision/core:frame_buffer",
        "//tensorflow_lite_support/cc/task/vision/proto:bounding_box_proto_inc",
        "//tensorflow_lite_support/cc/task/vision/utils:frame_buffer_utils",
        "@com_google_absl//absl/memory",
    ],
)

cc_library_with_tflite(
    name = "classification_postprocessor",
    srcs = ["classification_postprocessor.cc"],
    hdrs = ["classification_postprocessor.h"],
    tflite_deps = [
        ":processor",
        "//tensorflow_lite_support/cc/task/core:tflite_engine",
    ],
    deps = [
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/task/core:classification_head",
        "//tensorflow_lite_support/cc/task/core:label_map_item",
        "//tensorflow_lite_support/cc/task/core:score_calibration",
        "//tensorflow_lite_support/cc/task/core:task_utils",
        "//tensorflow_lite_support/cc/task/processor/proto:class_cc_proto",
        "//tensorflow_lite_support/cc/task/processor/proto:classification_options_cc_proto",
        "//tensorflow_lite_support/cc/task/processor/proto:classifications_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@org_tensorflow//tensorflow/lite/c:c_api_types",
    ],
)

cc_library_with_tflite(
    name = "embedding_postprocessor",
    srcs = ["embedding_postprocessor.cc"],
    hdrs = ["embedding_postprocessor.h"],
    tflite_deps = [
        ":processor",
        "//tensorflow_lite_support/cc/task/core:tflite_engine",
    ],
    deps = [
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/processor/proto:embedding_options_cc_proto",
        "@com_google_absl//absl/status",
    ],
)
