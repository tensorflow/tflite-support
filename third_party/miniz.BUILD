package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "libminiz",
    srcs = glob(["*.c"]),
    copts = [],
    features = [
        "-layering_check",
    ],
    textual_hdrs = glob(["*.h"]),
)
