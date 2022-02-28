# Placeholder for internal Python strict test compatibility macro.

package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

py_test(
    name = "metadata_loader_test",
    srcs = ["metadata_loader_test.py"],
    data = [
        "//tensorflow_lite_support/python/test/model_maker/core/data_util/testdata:searcher_images",
        "//tensorflow_lite_support/python/test/model_maker/core/data_util/testdata:searcher_images_metadata",
    ],
    deps = [
        "//tensorflow_lite_support/python/model_maker/core/data_util:metadata_loader",
        "//tensorflow_lite_support/python/test:test_util",
    ],
)

py_test(
    name = "searcher_dataloader_test",
    srcs = ["searcher_dataloader_test.py"],
    data = [
        "//tensorflow_lite_support/cc/test/testdata/task/vision:test_models",
    ],
    deps = [
        # build rule placeholder: numpy dep,
        "//tensorflow_lite_support/python/model_maker/core/data_util:searcher_dataloader",
        "//tensorflow_lite_support/python/task/core:task_options",
        "//tensorflow_lite_support/python/task/vision:image_embedder",
        "//tensorflow_lite_support/python/test:test_util",
    ],
)

py_test(
    name = "image_searcher_dataloader_test",
    srcs = ["image_searcher_dataloader_test.py"],
    data = [
        "//tensorflow_lite_support/cc/test/testdata/task/vision:test_models",
        "//tensorflow_lite_support/python/test/model_maker/core/data_util/testdata:searcher_images",
    ],
    deps = [
        "//tensorflow_lite_support/python/model_maker/core/data_util:image_searcher_dataloader",
        "//tensorflow_lite_support/python/test:test_util",
    ],
)
