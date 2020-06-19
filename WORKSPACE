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
    sha256 = "bb8b10da8184ce747f0348ea5b0d0aaf9e9bbe63cf68363d0e1bcdb72b4d3315",
    strip_prefix = "tensorflow-5d49dc5526324443931a33cc84d66c8bcae9cea2",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/5d49dc5526324443931a33cc84d66c8bcae9cea2.zip",  # 2020-06-13
    ],
)

# Set up dependencies. Need to do this before set up TF so that our modification
# could take effects.
load("//third_party:repo.bzl", "third_party_http_archive")

third_party_http_archive(
    name = "pybind11",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
        "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
    ],
    sha256 = "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d",
    strip_prefix = "pybind11-2.4.3",
    build_file = "//third_party:pybind11.BUILD",
)

http_archive(
    name = "absl_py",
    sha256 = "603febc9b95a8f2979a7bdb77d2f5e4d9b30d4e0d59579f88eba67d4e4cc5462",
    strip_prefix = "abseil-py-pypi-v0.9.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-py/archive/pypi-v0.9.0.tar.gz",
        "https://github.com/abseil/abseil-py/archive/pypi-v0.9.0.tar.gz",
    ],
)

http_archive(
    name = "com_google_absl",
    build_file = "//third_party:com_google_absl.BUILD",
    sha256 = "f368a8476f4e2e0eccf8a7318b98dafbe30b2600f4e3cf52636e5eb145aba06a",
    strip_prefix = "abseil-cpp-df3ea785d8c30a9503321a3d35ee7d35808f190d",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/df3ea785d8c30a9503321a3d35ee7d35808f190d.tar.gz",
    ],
)

http_archive(
    name = "six_archive",
    build_file = "//third_party:six.BUILD",
    sha256 = "d16a0141ec1a18405cd4ce8b4613101da75da0e9a7aec5bdd4fa804d0e0eba73",
    strip_prefix = "six-1.12.0",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.12.0.tar.gz",
    ],
)

load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")

flatbuffers()


# Set up TF.
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")
tf_workspace(tf_repo_name="@org_tensorflow")

load("//third_party/tensorflow:tf_configure.bzl", "tf_configure")
tf_configure(name = "local_config_tf")

# TF submodule compilation doesn't take care of grpc deps. Do it manually here.
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)
apple_rules_dependencies()

load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)
apple_support_dependencies()

load("@upb//bazel:repository_defs.bzl", "bazel_version_repository")
bazel_version_repository(name = "bazel_version")


# Set up Android.
load("//third_party/android:android_configure.bzl", "android_configure")
android_configure(name="local_config_android")
load("@local_config_android//:android.bzl", "android_workspace")
android_workspace()

