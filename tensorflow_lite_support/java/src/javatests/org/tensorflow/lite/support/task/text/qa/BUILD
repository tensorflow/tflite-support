load("@build_bazel_rules_android//android:rules.bzl", "android_library", "android_local_test")

package(
    default_testonly = 1,
)

licenses(["notice"])  # Apache License 2.0

android_library(
    name = "test_assets",
    assets = [
        "//third_party/tensorflow_lite_support/cc/testdata:albert_model",
        "//third_party/tensorflow_lite_support/cc/testdata:mobile_bert_model",
    ],
    assets_dir = "text",
    manifest = "AndroidManifest.xml",
)

android_local_test(
    name = "BertQuestionAnswererTest",
    timeout = "long",
    srcs = ["BertQuestionAnswererTest.java"],
    nocompress_extensions = [
        "tflite",
        "txt",
        "model",
    ],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "optonly",  # The test takes long, and only run with -c opt.
    ],
    test_class = "org.tensorflow.lite.support.task.text.qa.BertQuestionAnswererTest",
    deps = [
        ":test_assets",
        "//third_party/java/junit:junit-android",
        "//third_party/java/truth:truth-android",
        "//third_party/tensorflow_lite_support/java/src/java/org/tensorflow/lite/support/task/text/qa:bert_question_answerer",
    ],
)
