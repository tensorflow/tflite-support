# Python Demos for Text Task APIs

A Python wrapper for the C++ Text Task APIs.

## Background
This Python API is based on the C++ Text Task APIs. It uses shared libraries which are built using `bazel`. To bridge C++ and Python APIs,  [ctypes](https://docs.python.org/3/library/ctypes.html) which is a foreign function library for Python is used.

## NLClassifier

#### Prerequisites

You will need:

* a TFLite text classification model with certain format.
(e.g. [movie_review_model][1], a model to classify movie reviews), you'll need
to configure the input tensor and out tensor for the API, see the [doc][2] for 
details.
* Shared library (.so) for NLClassifier
#### Usage

In the console, run:

```bash
# Download the model:
curl \
 -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite' \
 -o /tmp/movie_review.tflite

# Generate the shared library. This is a one time task
bazel build -c opt tensorflow_lite_support/examples/task/text/desktop/python/cc:invoke_nl_classifier

# Run the detection tool:
python tensorflow_lite_support/examples/task/text/desktop/python/nl_classifier_demo.py \
 --model_path=/tmp/movie_review.tflite \
 --text="What a waste of my time." \
 --input_tensor_name="input_text" \
 --output_score_tensor_name="probability"
```

#### Results

In the console, you should get:

```
category[0]: 'Negative' : '0.81313'
category[1]: 'Positive' : '0.18687'
```

[1]: https://www.tensorflow.org/lite/models/text_classification/overview
[2]: https://github.com/tensorflow/tflite-support/blob/fe8b69002f5416900285dc69e2baa078c91bd994/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h#L55