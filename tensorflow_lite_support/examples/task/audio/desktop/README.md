# CLI Demos for C++ Audio Task APIs

This folder contains simple command-line tools for easily trying out the C++
Audio Task APIs.

## Coral integration

Task Library now supports fast TFLite inference delegated onto
[Coral Edge TPU devices](https://coral.ai/docs/edgetpu/inference/). See the
[documentation](https://www.tensorflow.org/lite/inference_with_metadata/task_library/overview#run_task_library_with_delegates)
for more details. To run the demo on a Coral device, add the following
configurations to the bazel command:

```bash
# On the Linux
CORAL_SETTING="--define darwinn_portable=1 --linkopt=-lusb-1.0"

# On the Mac
# add '--linkopt=-lusb-1.0 --linkopt=-L/opt/local/lib/' if you are
# using MacPorts or '--linkopt=-lusb-1.0 --linkopt=-L/opt/homebrew/lib' if you
# are using Homebrew.
CORAL_SETTING="--define darwinn_portable=1 --linkopt=-L/opt/local/lib/ --linkopt=-lusb-1.0"

# Windows is not supported yet.
```

Note, the `libusb-1.0-0-dev` package is required. It can be installed as
follows:

```bash
# On the Linux
sudo apt-get install libusb-1.0-0-dev

# On the macOS
port install libusb
# or
brew install libusb
```

See the example commands in each task demo below.

You can also explore more [pretrained Coral model](https://coral.ai/models) and
try them in the demo. All the models have populated with
[TFLite Model Metadata](https://www.tensorflow.org/lite/convert/metadata).

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

# Download the audio file:
curl \
 -L https://storage.googleapis.com/audioset/miaow_16k.wav \
 -o /tmp/miao.wav

# Run the classification tool:
bazel run -c opt \
 tensorflow_lite_support/examples/task/audio/desktop:audio_classifier_demo -- \
  --model_path=/tmp/yamnet.tflite \
  --score_threshold=0.5 \
  --audio_wav_path=/tmp/miao.wav
```

To run the demo on a [Coral Edge TPU device](https://coral.ai/products/), create
the Coral configurations, `CORAL_SETTING` (see the section,
[Coral integration](#coral-integration)), then run:

```bash
# Download the Coral model:
curl \
 -L 'https://tfhub.dev/google/coral-model/yamnet/classification/coral/1?coral-format=tflite' \
 -o /tmp/yamnet_edgetpu.tflite

# Run the classification tool:
bazel run -c opt ${CORAL_SETTING} \
 tensorflow_lite_support/examples/task/audio/desktop:audio_classifier_demo -- \
  --model_path=/tmp/yamnet_edgetpu.tflite \
  --audio_wav_path=/path/to/the/audio_file.wav \
  --score_threshold=0.5 \
  --use_coral=true
```

### Results
In the console, you should get:

```bash
Time cost to classify the input audio clip on CPU: 51.4087 ms
Note: Only showing classes with score higher than 0.5

Head[0]: scores
	category[Cat]: 0.73828
	category[Animal]: 0.66797
	category[Domestic animals, pets]: 0.66797
```
