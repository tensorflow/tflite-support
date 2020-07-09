load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_portable_proto_library")
load(
    "@org_tensorflow//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    default_visibility = [
        "//tensorflow_lite_support:users",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Common vision protos.

tf_proto_library(
    name = "bounding_box_proto",
    srcs = ["bounding_box.proto"],
    cc_api_version = 2,
)

tf_portable_proto_library(
    name = "bounding_box_portable_proto",
    config_string = "allow_all:true",
    header_outs = ["//tensorflow_lite_support/cc/task/vision/proto/bounding_box.proto.h"],
    proto_deps = [
        ":bounding_box_proto",
    ],
)

cc_library(
    name = "bounding_box_proto_inc",
    hdrs = ["bounding_box_proto_inc.h"],
    deps = [":bounding_box_portable_proto"],
)

tf_proto_library(
    name = "class_proto",
    srcs = ["class.proto"],
    cc_api_version = 2,
)

tf_portable_proto_library(
    name = "class_portable_proto",
    config_string = "allow_all:true",
    header_outs = ["//tensorflow_lite_support/cc/task/vision/proto/class.proto.h"],
    proto_deps = [
        ":class_proto",
    ],
)

cc_library(
    name = "class_proto_inc",
    hdrs = ["class_proto_inc.h"],
    deps = [":class_portable_proto"],
)

# ObjectDetector protos.

tf_proto_library(
    name = "object_detector_options_proto",
    srcs = ["object_detector_options.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto",
    ],
)

tf_portable_proto_library(
    name = "object_detector_options_portable_proto",
    config_string = "allow_all:true",
    header_outs = ["//tensorflow_lite_support/cc/task/vision/proto/object_detector_options.proto.h"],
    portable_deps = [
        "//tensorflow_lite_support/cc/task/core/proto:external_file_portable_proto",
    ],
    proto_deps = [
        ":object_detector_options_proto",
    ],
)

cc_library(
    name = "object_detector_options_proto_inc",
    hdrs = ["object_detector_options_proto_inc.h"],
    deps = [
        ":object_detector_options_portable_proto",
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto_inc",
    ],
)

tf_proto_library(
    name = "detections_proto",
    srcs = ["detections.proto"],
    cc_api_version = 2,
    protodeps = [
        ":bounding_box_proto",
        ":class_proto",
    ],
)

tf_portable_proto_library(
    name = "detections_portable_proto",
    config_string = "allow_all:true",
    header_outs = ["//tensorflow_lite_support/cc/task/vision/proto/detections.proto.h"],
    portable_deps = [
        ":class_portable_proto",
        ":bounding_box_portable_proto",
    ],
    proto_deps = [
        ":detections_proto",
    ],
)

cc_library(
    name = "detections_proto_inc",
    hdrs = ["detections_proto_inc.h"],
    deps = [
        ":bounding_box_proto_inc",
        ":class_proto_inc",
        ":detections_portable_proto",
    ],
)

# ImageClassifier protos.

tf_proto_library(
    name = "image_classifier_options_proto",
    srcs = ["image_classifier_options.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto",
    ],
)

tf_portable_proto_library(
    name = "image_classifier_options_portable_proto",
    config_string = "allow_all:true",
    header_outs = ["//tensorflow_lite_support/cc/task/vision/proto/image_classifier_options.proto.h"],
    portable_deps = [
        "//tensorflow_lite_support/cc/task/core/proto:external_file_portable_proto",
    ],
    proto_deps = [
        ":image_classifier_options_proto",
    ],
)

cc_library(
    name = "image_classifier_options_proto_inc",
    hdrs = ["image_classifier_options_proto_inc.h"],
    deps = [
        ":image_classifier_options_portable_proto",
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto_inc",
    ],
)

tf_proto_library(
    name = "classifications_proto",
    srcs = ["classifications.proto"],
    cc_api_version = 2,
    protodeps = [
        ":class_proto",
    ],
)

tf_portable_proto_library(
    name = "classifications_portable_proto",
    config_string = "allow_all:true",
    header_outs = ["//tensorflow_lite_support/cc/task/vision/proto/classifications.proto.h"],
    portable_deps = [
        ":class_portable_proto",
    ],
    proto_deps = [
        ":classifications_proto",
    ],
)

cc_library(
    name = "classifications_proto_inc",
    hdrs = ["classifications_proto_inc.h"],
    deps = [
        ":class_proto_inc",
        ":classifications_portable_proto",
    ],
)

# ImageSegmenter protos.

tf_proto_library(
    name = "image_segmenter_options_proto",
    srcs = ["image_segmenter_options.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto",
    ],
)

tf_portable_proto_library(
    name = "image_segmenter_options_portable_proto",
    config_string = "allow_all:true",
    header_outs = ["//tensorflow_lite_support/cc/task/vision/proto/image_segmenter_options.proto.h"],
    portable_deps = [
        "//tensorflow_lite_support/cc/task/core/proto:external_file_portable_proto",
    ],
    proto_deps = [
        ":image_segmenter_options_proto",
    ],
)

cc_library(
    name = "image_segmenter_options_proto_inc",
    hdrs = ["image_segmenter_options_proto_inc.h"],
    deps = [
        ":image_segmenter_options_portable_proto",
        "//tensorflow_lite_support/cc/task/core/proto:external_file_proto_inc",
    ],
)

tf_proto_library(
    name = "segmentations_proto",
    srcs = ["segmentations.proto"],
    cc_api_version = 2,
)

tf_portable_proto_library(
    name = "segmentations_portable_proto",
    config_string = "allow_all:true",
    header_outs = ["//tensorflow_lite_support/cc/task/vision/proto/segmentations.proto.h"],
    proto_deps = [
        ":segmentations_proto",
    ],
)

cc_library(
    name = "segmentations_proto_inc",
    hdrs = ["segmentations_proto_inc.h"],
    deps = [":segmentations_portable_proto"],
)
