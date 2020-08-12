# Place holder for py clif macro.

package(
    default_visibility = [
        "//tensorflow_lite_support:users",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_clif_cc(
    name = "model_converter",
    srcs = [
        "model_converter.clif",
    ],
    deps = [
        "//tensorflow_lite_support/custom_ops/kernel/sentencepiece:model_converter",
    ],
)
