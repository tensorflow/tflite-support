# CLI Demos for Python Text Task APIs

A Python wrapper for the C++ Text Task APIs.

## Background
This Python API is based on the C++ Text Task APIs. It uses Python's
[subprocess](https://docs.python.org/3/library/subprocess.html)
to call C++ Text Task APIs.

## NLClassifier

#### Prerequisites

You will need:

* a TFLite text classification model with certain format.
(e.g. [movie_review_model][1], a model to classify movie reviews).

#### Usage

In the console, run:

```bash
# Download the model:
curl \
 -L 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/text_classification/text_classification_v2.tflite' \
 -o /tmp/movie_review.tflite

# Run the detection tool:
python \
 tensorflow_lite_support/examples/task/text/desktop/python/nl_classifier_demo.py \
 --model_path=/tmp/movie_review.tflite \
 --text="What a waste of my time."
```

#### Results

In the console, you should get:

```
Time cost to classify the input text on CPU: 0.088 ms
category[0]: 'Negative' : '0.81313'
category[1]: 'Positive' : '0.18687'
```

[1]: https://www.tensorflow.org/lite/models/text_classification/overview
[2]: https://github.com/tensorflow/tflite-support/blob/fe8b69002f5416900285dc69e2baa078c91bd994/tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h#L55