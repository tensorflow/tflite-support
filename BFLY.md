# Metadata on TFLITE models on the phone

We use this library to generate a static framework for model version. At some point we may want to
expand on this as a bunch of information can be pulled from the ModelMetadata stored in tflite models.
But for now...

The library we're adapting is the build target

    //tensorflow_lite_support/metadata/cc:metadata_extractor

The header for this library has a number of awkward imports given our current DLTools podspec. 
So wrapped this in a target of our own

    //tensorflow_lite_support/metadata/cc:dl_tflite_metadata

that exports a single function for model version.  Delegating to the 'metatdata_extractor' 
implementation resulted in a iOS binary that was still 700K larger than the unversioned phone
code (Outrageous!).  So instead of calling through to the 
metadata_extractor, we just pull the relevant code, thinned, into the `dl_tflite_metadata` class. 
This version increases the binary size by 36K, which seems reasonable if not dainty.

Finally, this is wrapped in a static ios framework build.

    tflite-support$ bazel build -c opt --config=ios_fat //tensorflow_lite_support/metadata/cc:TfliteMetadata_framework

