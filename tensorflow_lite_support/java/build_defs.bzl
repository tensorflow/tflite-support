"""Configures how Java libraries should be built in TFLite Support."""

# Forked from TF Java build_defs.bzl, but add more options for TFLS.

load(
    "@org_tensorflow//tensorflow/java:build_defs.bzl",
    TF_EP_ENABLED_WARNINGS = "EP_ENABLED_WARNINGS",
    TF_JAVA_VERSION_OPTS = "JAVA_VERSION_OPTS",
    TF_XLINT_OPTS = "XLINT_OPTS",
)

# We need "release 7" to resolve ByteBuffer incompatibility in Java versions.
# https://www.morling.dev/blog/bytebuffer-and-the-dreaded-nosuchmethoderror/

JAVA_VERSION_OPTS = TF_JAVA_VERSION_OPTS + ["--release 7"]

JAVACOPTS = TF_JAVA_VERSION_OPTS + TF_XLINT_OPTS + TF_EP_ENABLED_WARNINGS
