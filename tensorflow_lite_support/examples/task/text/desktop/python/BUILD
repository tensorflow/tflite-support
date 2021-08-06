package(
    default_visibility = [
        "//tensorflow_lite_support:users",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# bazel run \
# tensorflow_lite_support/examples/task/text/desktop/python:nl_classifier_demo \
# -- \
# --model_path=/path/to/model.tflite \
# --text="What a waste of my time."
py_binary(
    name = "nl_classifier_demo",
    srcs = ["nl_classifier_demo.py"],
    data = ["//tensorflow_lite_support/examples/task/text/desktop:nl_classifier_demo"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

# bazel run \
# tensorflow_lite_support/examples/task/text/desktop/python:bert_nl_classifier_demo \
# -- \
# --model_path=/path/to/model.tflite \
# --text="it's a charming and often affecting journey"
py_binary(
    name = "bert_nl_classifier_demo",
    srcs = ["bert_nl_classifier_demo.py"],
    data = ["//tensorflow_lite_support/examples/task/text/desktop:bert_nl_classifier_demo"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
