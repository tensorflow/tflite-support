load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//tensorflow_lite_support:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "image_embedder",
    srcs = [
        "image_embedder.cc",
    ],
    module_name = "image_embedder",
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/vision:image_embedder",
        "//tensorflow_lite_support/examples/task/vision/desktop/utils:image_utils",
        "@org_tensorflow//tensorflow/lite/kernels:reference_ops",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

pybind_extension(
    name = "image_classifier",
    srcs = [
        "image_classifier.cc",
    ],
    module_name = "image_classifier",
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/vision:image_classifier",
        "//tensorflow_lite_support/examples/task/vision/desktop/utils:image_utils",
        "@org_tensorflow//tensorflow/lite/kernels:reference_ops",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)
