# Placeholder for internal Python strict library compatibility macro.

package(
    default_visibility = ["//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "audio_embedder",
    srcs = [
        "audio_embedder.py",
    ],
    deps = [
        "//tensorflow_lite_support/python/task/core/proto:base_options_py_pb2",
        "//tensorflow_lite_support/python/task/processor/proto:embedding_options_pb2",
        "//tensorflow_lite_support/python/task/processor/proto:embeddings_pb2",
        "//tensorflow_lite_support/python/task/audio/core:tensor_audio",
        "//tensorflow_lite_support/python/task/audio/core/pybinds:_pywrap_audio_buffer",
        "//tensorflow_lite_support/python/task/audio/pybinds:_pywrap_audio_embedder",
        "//tensorflow_lite_support/python/task/audio/pybinds:audio_embedder_options_pb2",
    ],
)
