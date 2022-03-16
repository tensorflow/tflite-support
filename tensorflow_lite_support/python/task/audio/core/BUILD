# Placeholder for internal Python strict test compatibility macro.
# Placeholder for internal Python strict library compatibility macro.

package(
    default_visibility = ["//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "audio_record",
    srcs = ["audio_record.py"],
    deps = [
        # build rule placeholder: numpy dep,
        # build rule placeholder: sounddevice dep
    ],
)

py_library(
    name = "tensor_audio",
    srcs = ["tensor_audio.py"],
    deps = [
        ":audio_record",
        # build rule placeholder: numpy dep,
        "//tensorflow_lite_support/python/task/audio/core/pybinds:_pywrap_audio_buffer",
    ],
)

py_test(
    name = "tensor_audio_test",
    srcs = ["tensor_audio_test.py"],
    data = [
        "//tensorflow_lite_support/cc/test/testdata/task/audio:test_audio_clips",
    ],
    deps = [
        ":tensor_audio",
        # build rule placeholder: numpy dep,
        # build rule placeholder: tensorflow dep,
        "//tensorflow_lite_support/python/task/audio/core/pybinds:_pywrap_audio_buffer",
        "//tensorflow_lite_support/python/test:test_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

