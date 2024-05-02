bazel build --copt=-w \
        --cpu=arm64-v8a \
        --cxxopt=-std=c++17 \
        --fat_apk_cpu=arm64-v8a \
        tensorflow_lite_support/java:tensorflowlite_support_java.aar