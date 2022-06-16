"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "flatbuffers",
        # TODO(b/235550563): Update Flatbuffers to 2.0.6.
        strip_prefix = "flatbuffers-2.0.5",
        sha256 = "b01e97c988c429e164c5c7df9e87c80007ca87f593c0d73733ba536ddcbc8f98",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v2.0.5.tar.gz",
            "https://github.com/google/flatbuffers/archive/v2.0.5.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        delete = ["build_defs.bzl", "BUILD.bazel"],
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
