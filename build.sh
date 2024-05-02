bazel build --copt=-w \
        --cpu=arm64-v8a \
        --cxxopt=-std=c++17 \
        --fat_apk_cpu=arm64-v8a \
        tensorflow_lite_support/java:tensorflowlite_support_java.aar

bazel build --fat_apk_cpu=arm64-v8a tensorflow_lite_support/odml/java/image
bazel build --fat_apk_cpu=arm64-v8a tensorflow_lite_support/metadata/java:tensorflow-lite-support-metadata-lib
bazel build --fat_apk_cpu=arm64-v8a //tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/core:base-task-api
bazel build --fat_apk_cpu=arm64-v8a //tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision:task-library-vision