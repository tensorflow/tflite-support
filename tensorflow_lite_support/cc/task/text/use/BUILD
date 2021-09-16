load(
    "@org_tensorflow//tensorflow/lite/core/shims:cc_library_with_tflite.bzl",
    "cc_library_with_tflite",
)

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library_with_tflite(
    name = "universal_sentence_encoder_qa",
    srcs = [
        "universal_sentence_encoder_qa.cc",
    ],
    hdrs = [
        "universal_sentence_encoder_qa.h",
    ],
    tflite_deps = [
        "@org_tensorflow//tensorflow/lite/core/shims:builtin_ops",
        "//tensorflow_lite_support/cc/task/core:base_task_api",
        "//tensorflow_lite_support/cc/task/core:task_api_factory",
        "//tensorflow_lite_support/cc/task/core:tflite_engine",
    ],
    deps = [
        "//tensorflow_lite_support/cc/port:status_macros",
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/core:task_utils",
        "//tensorflow_lite_support/cc/task/text/proto:retrieval_cc_proto",
        "//tensorflow_lite_support/cc/task/vision/proto:embeddings_cc_proto",
        "//tensorflow_lite_support/cc/text/tokenizers:bert_tokenizer",
        "//tensorflow_lite_support/cc/text/tokenizers:sentencepiece_tokenizer",
        "//tensorflow_lite_support/cc/text/tokenizers:tokenizer",
        "//tensorflow_lite_support/cc/text/tokenizers:tokenizer_utils",
        "//tensorflow_lite_support/metadata:metadata_schema_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)
