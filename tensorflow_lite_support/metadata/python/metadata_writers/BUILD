# Placeholder for internal Python strict library compatibility macro.

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "metadata_writer",
    srcs = [
        "metadata_writer.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":writer_utils",
        "//tensorflow_lite_support/metadata:metadata_schema_py",
        "//tensorflow_lite_support/metadata:schema_py",
        "//tensorflow_lite_support/metadata/python:metadata",
        "@flatbuffers//:runtime_py",
    ],
)

py_library(
    name = "metadata_info",
    srcs = [
        "metadata_info.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_lite_support/metadata:metadata_schema_py",
        "//tensorflow_lite_support/metadata:schema_py",
    ],
)

py_library(
    name = "writer_utils",
    srcs = [
        "writer_utils.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow_lite_support/metadata:metadata_schema_py",
        "//tensorflow_lite_support/metadata:schema_py",
    ],
)

py_library(
    name = "image_classifier",
    srcs = [
        "image_classifier.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
        "//tensorflow_lite_support/metadata:metadata_schema_py",
    ],
)

py_library(
    name = "object_detector",
    srcs = [
        "object_detector.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
        "//tensorflow_lite_support/metadata:metadata_schema_py",
        "//tensorflow_lite_support/metadata/python:metadata",
        "@flatbuffers//:runtime_py",
    ],
)

py_library(
    name = "image_segmenter",
    srcs = [
        "image_segmenter.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
        "//tensorflow_lite_support/metadata:metadata_schema_py",
    ],
)

py_library(
    name = "nl_classifier",
    srcs = [
        "nl_classifier.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
    ],
)

py_library(
    name = "audio_classifier",
    srcs = [
        "audio_classifier.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":metadata_info",
        ":metadata_writer",
        ":writer_utils",
    ],
)

py_library(
    name = "metadata_writer_for_qa_lib",
    srcs = ["metadata_writer_for_qa.py"],
    srcs_version = "PY3",
    deps = [
        # build rule placeholder: tensorflow dep,
        "//tensorflow_lite_support/metadata/python/metadata_writers:metadata_info",
        "//tensorflow_lite_support/metadata/python/metadata_writers:metadata_writer",
        "//tensorflow_lite_support/metadata/python/metadata_writers:writer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "metadata_writer_for_qa",
    srcs = ["metadata_writer_for_qa.py"],
    python_version = "PY3",
    deps = [
        ":metadata_writer_for_qa_lib",
    ],
)
