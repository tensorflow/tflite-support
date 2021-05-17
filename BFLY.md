# Metadata on TFLITE models on the phone

We use this library to generate a static framework for model version. At some point we may want to
expand on this as a bunch of information can be pulled from the ModelMetadata stored in tflite models.
But for now...

The library we're adapting is the build target

    //tensorflow_lite_support/metadata/cc:metadata_extractor

The header for this library has a number of awkward imports given our current DLTools podspec. So we've
built a narrow library:

    //tensorflow_lite_support/metadata/cc:dl_tflite_metadata

that exports a single function for model version for now.

Finally, this is wrapped in a static framework build. This looks large, but thins down on final binary build.

    tflite-support$ bazel build -c opt --config=ios_fat //tensorflow_lite_support/metadata/cc:TfliteMetadata_framework

