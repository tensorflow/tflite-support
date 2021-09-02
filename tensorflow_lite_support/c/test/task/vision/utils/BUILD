package(
    default_visibility = [
        "//tensorflow_lite_support:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "image_utils",
    srcs = ["image_utils.c"],
    hdrs = ["image_utils.h"],
    deps = [
        "@stblib//:stb_image",
        "@stblib//:stb_image_write",
    ],
)
