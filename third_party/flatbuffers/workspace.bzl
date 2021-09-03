"""Loads the Flatbuffers library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "flatbuffers",
        strip_prefix = "flatbuffers-2.0.0",
        sha256 = "9ddb9031798f4f8754d00fca2f1a68ecf9d0f83dfac7239af1311e4fd9a565c4",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v2.0.0.tar.gz",
            "https://github.com/google/flatbuffers/archive/v2.0.0.tar.gz",
        ],
        build_file = "//third_party/flatbuffers:BUILD.bazel",
        delete = ["build_defs.bzl"],
        link_files = {
            "//third_party/flatbuffers:build_defs.bzl": "build_defs.bzl",
        },
    )
