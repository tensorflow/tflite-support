load("@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library_with_tflite(
    name = "image_classifier_c_api",
    srcs = [
        "image_classifier.cc",
    ],
    hdrs = [
        "image_classifier.h",
        "//tensorflow_lite_support/c/task/processor:classification_result_c_api",
        "//tensorflow_lite_support/c/task/processor:bounding_box"
    ],
    tflite_deps = [
        "//tensorflow_lite_support/cc/task/vision:image_classifier",
    ],
    visibility = ["//tensorflow_lite_support:__subpackages__"],
    deps = [
        "//tensorflow_lite_support/c/task/vision/utils:frame_buffer_cpp_c_utils",
        "//tensorflow_lite_support/c/task/processor:classification_result_c_api",
        "//tensorflow_lite_support/c/task/processor:bounding_box",
    ],
)

