load("@org_tensorflow//tensorflow:tensorflow.bzl", "pybind_extension")
# Placeholder for internal Python strict library compatibility macro.

package(
    default_visibility = [
        "//tensorflow_lite_support:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

pybind_extension(
    name = "_pywrap_audio_classifier",
    srcs = [
        "_pywrap_audio_classifier.cc",
    ],
    module_name = "_pywrap_audio_classifier",
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/processor/proto:classification_options_cc_proto",
        "//tensorflow_lite_support/cc/task/audio:audio_classifier",
        "//tensorflow_lite_support/python/task/core/pybinds:task_utils",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)
