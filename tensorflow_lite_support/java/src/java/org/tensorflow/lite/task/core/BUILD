load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@org_tensorflow//tensorflow/java:build_defs.bzl", "JAVACOPTS")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

android_library(
    name = "base-task-api",
    srcs = glob(["**/*.java"]),
    javacopts = JAVACOPTS,
    visibility = ["//visibility:public"],
    # LINT.IfChange(dep)
    deps = [
        "@com_google_auto_value",
    ],
    # LINT.ThenChange(<INTERNAL>/release/build_task_base_pom.sh:dep)
)

alias(
    name = "base_task_api",
    actual = ":base-task-api",
)
