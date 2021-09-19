load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    default_visibility = [
        "//tensorflow_lite_support:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library_with_tflite(
    name = "image_classifier",
    srcs = [
        "image_classifier.cc",
    ],
    hdrs = [
        "image_classifier.h",
    ],
    tflite_deps = [
        "//tensorflow_lite_support/cc/task/vision:image_classifier",
    ],
    deps = [
        "//tensorflow_lite_support/c/task/core:base_options",
        "//tensorflow_lite_support/c/task/processor:bounding_box",
        "//tensorflow_lite_support/c/task/processor:classification_options",
        "//tensorflow_lite_support/c/task/processor:classification_result",
        "//tensorflow_lite_support/c/task/vision/core:frame_buffer",
        "//tensorflow_lite_support/c/task/vision/utils:frame_buffer_cpp_c_utils",
        "//tensorflow_lite_support/cc/task/vision/proto:classifications_proto_inc",
        "//tensorflow_lite_support/cc/task/vision/proto:image_classifier_options_proto_inc",
    ],
)
