package(
    default_visibility = ["//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "AVAudioPCMBufferUtils",
    srcs = [
        "sources/AVAudioPCMBuffer+Utils.m",
    ],
    hdrs = [
        "sources/AVAudioPCMBuffer+Utils.h",
    ],
    module_name = "AVAudioPCMBufferUtils",
    sdk_frameworks = ["AVFoundation"],
    deps = [
        "//third_party/apple_frameworks:AVFoundation",
        "//third_party/apple_frameworks:Foundation",
    ],
)
