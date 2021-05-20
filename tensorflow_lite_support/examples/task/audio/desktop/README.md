# CLI Demos for C++ Audio Task APIs

This folder contains simple command-line tools for easily trying out the C++
Audio Task APIs.

## Audio Classification

### Prerequisites
You will need:

-   a TFLite audio classification model with metadata (e.g.
    https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1, an
    environmental sound classification model available on TensorFlow Hub),
-   a mono-channel 16-bit PCM WAV file. The sample rate of the WAV file should
    be the same as what model requires (described in the Metadata).

#### Usage

In the console, run:

```bash
# Download the model:
curl \
 -L 'https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite' \
 -o /tmp/yamnet.tflite

# Run the detection tool:
bazel run -c opt \
 tensorflow_lite_support/examples/task/audio/desktop:audio_classifier_demo -- \
  --model_path=/tmp/yamnet.tflite \
  --audio_wav_path=/path/to/the/audio_file.wav
```

### Results
In the console, you should get:

```bash
Head[0]: yamnet_classification
        category[Speech]: 0.99129
        category[Inside, small room]: 0.04351
        category[Conversation]: 0.00642
        category[Narration, monologue]: 0.00388
        category[Speech synthesizer]: 0.00198
        category[Inside, large room or hall]: 0.00176
```

