load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
# Placeholder for internal Python strict library compatibility macro.

package(
    default_visibility = [
        "//tensorflow_lite_support:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "audio_embedder_options_pb2",
    srcs = ["audio_embedder_options_pb2.py"],
    deps = ["//tensorflow_lite_support/cc/task/audio/proto:audio_embedder_options_py_pb2"],
)

pybind_extension(
    name = "_pywrap_audio_embedder",
    srcs = [
        "_pywrap_audio_embedder.cc",
    ],
    module_name = "_pywrap_audio_embedder",
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/python/task/core/pybinds:task_utils",
        "//tensorflow_lite_support/cc/task/processor/proto:embedding_options_cc_proto",
        "//tensorflow_lite_support/cc/task/audio:audio_embedder",
        "//tensorflow_lite_support/cc/task/audio/core:audio_buffer",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)
