load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

package(
    default_visibility = ["//tensorflow_lite_support:users"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "audio_classifier_native",
    srcs = [
        ":libtask_audio_jni.so",
    ],
)

tflite_jni_binary(
    name = "libtask_audio_jni.so",
    srcs = [
        "audio_classifier_jni.cc",
    ],
    linkscript = "//tensorflow_lite_support/java:default_version_script.lds",
    deps = [
        "//tensorflow_lite_support/cc/port:statusor",
        "//tensorflow_lite_support/cc/task/audio:audio_classifier",
        "//tensorflow_lite_support/cc/task/audio/proto:audio_classifier_options_cc_proto",
        "//tensorflow_lite_support/cc/task/audio/proto:classifications_proto_inc",
        "//tensorflow_lite_support/cc/utils:jni_utils",
        "//tensorflow_lite_support/java/jni",
    ],
)
