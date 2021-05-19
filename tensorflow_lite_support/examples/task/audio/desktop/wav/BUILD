# Description:
# wav_io targets.

package(
    default_visibility = [
        "//tensorflow_lite_support:users",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "wav_io",
    srcs = [
        "wav_io.cc",
    ],
    hdrs = ["wav_io.h"],
    deps = [
        "//tensorflow_lite_support/cc/port:status_macros",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)
