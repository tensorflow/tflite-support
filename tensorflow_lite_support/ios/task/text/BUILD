package(
    default_visibility = ["//tensorflow_lite_support:internal"],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "TFLTextSearcher",
    srcs = [
        "sources/TFLTextSearcher.mm",
    ],
    hdrs = [
        "sources/TFLTextSearcher.h",
    ],
    copts = [
        "-ObjC++",
        "-std=c++17",
    ],
    features = ["-layering_check"],
    module_name = "TFLTextSearcher",
    deps = [
        "//tensorflow_lite_support/cc/task/text:text_searcher",
        "//tensorflow_lite_support/ios:TFLCommonUtils",
        "//tensorflow_lite_support/ios/task/core:TFLBaseOptionsCppHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLEmbeddingOptionsHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLSearchOptionsHelpers",
        "//tensorflow_lite_support/ios/task/processor:TFLSearchResultHelpers",
    ],
)
