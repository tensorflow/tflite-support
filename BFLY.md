# Metadata on TFLITE models on the phone

We use this library to generate a compact framework for model version. At some point we may want to
expand on this as a wealth of information can be pulled from the ModelMetadata stored in tflite models.
But for now...

The library we're adapting is the build target

    //tensorflow_lite_support/metadata/cc:metadata_extractor

The header for this library has a number of awkward imports given our current DLTools podspec. So we've
built a narrow library:

    //tensorflow_lite_support/metadata/cc:dl_tflite_metadata

that exports a single function for model version for now.

We need libraries for both arm64 (iOS) and x86_64 (Simulator). For some reason the `ios_fat` target defined
int the `.bazelrc` file doesn't seem to work, so for now building this in "parts" as follows:

    tflite-support$ bazel build --config=ios_arm64 //tensorflow_lite_support/metadata/cc:dl_tflite_metadata
    tflite-support$ bazel build --config=ios_x86_64 //tensorflow_lite_support/metadata/cc:dl_tflite_metadata

We use a cc_binary target that ls `linkshared` so that we resolve a number of static linked references that
would be undefined on the phone. These both generate libraries with the same name (`libdl_tflite_metadata.so`)
so they were renamed with an appropriate suffix indicating their architecture after the build.

After this, they were combined into a "fat" binary as follows:

    ~$ lipo -create libdl_tflite_metadata_arm64.so libdl_tflite_metadata_x86_64.so -output TfliteSupport
    ~$ file TfliteSupport
    TfliteSupport: Mach-O universal binary with 2 architectures: [x86_64:Mach-O 64-bit dynamically linked shared library x86_64] [arm64]
    TfliteSupport (for architecture x86_64):	Mach-O 64-bit dynamically linked shared library x86_64
    TfliteSupport (for architecture arm64):	Mach-O 64-bit dynamically linked shared library arm64


