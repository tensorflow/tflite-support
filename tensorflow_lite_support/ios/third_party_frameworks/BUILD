package(
    default_visibility = ["//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)
load("@build_bazel_rules_apple//apple:apple.bzl", "apple_static_framework_import")

apple_static_framework_import(
    name = "MLImage",
    framework_imports = glob(["MLImage.framework/**"]),
)
