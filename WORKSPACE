workspace(name = "org_tensorflow_lite_support")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "org_tensorflow",
    strip_prefix = "tensorflow-2.2.0",
    sha256 = "fd3e6580cfe2035aa80d569b76bba5f33119362907f3d77039b6bedf76172712",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v2.2.0.zip"
    ],
)

# Configure TF.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(tf_repo_name="@org_tensorflow")

load("//third_party/tensorflow:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")


# Configure Android.
load("//third_party/android:android_configure.bzl", "android_configure")

android_configure(name="local_config_android")

load("@local_config_android//:android.bzl", "android_workspace")

android_workspace()
