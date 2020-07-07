#TFLite Task library - C++

A flexible and ready-to-use library for common machine learning model types,
such as classification and detection.

## Text Task Librarys

### QuestionAnswerer

`QuestionAnswerer` API is able to load
[Mobile BERT](https://tfhub.dev/tensorflow/mobilebert/1) or
[AlBert](https://tfhub.dev/tensorflow/albert_lite_base/1) TFLite models and
answer question based on context.

Use the C++ API to answer questions as follows:

```cc
using tflite::support::task::text::qa::BertQuestionAnswerer;
using tflite::support::task::text::qa::QaAnswer;
// Create API handler with Mobile Bert model.
auto qa_client = BertQuestionAnswerer::CreateBertQuestionAnswerer("/path/to/mobileBertModel", "/path/to/vocab");
// Or create API handler with Albert model.
// auto qa_client = BertQuestionAnswerer::CreateAlbertQuestionAnswerer("/path/to/alBertModel", "/path/to/sentencePieceModel");


std::string context =
    "Nikola Tesla (Serbian Cyrillic: Никола Тесла; 10 "
    "July 1856 – 7 January 1943) was a Serbian American inventor, electrical "
    "engineer, mechanical engineer, physicist, and futurist best known for his "
    "contributions to the design of the modern alternating current (AC) "
    "electricity supply system.";
std::string question = "When was Nikola Tesla born?";
// Run inference with `context` and a given `question` to the context, and get top-k
// answers ranked by logits.
const std::vector<QaAnswer> answers = qa_client->Answer(context, question);
// Access QaAnswer results.
for (const QaAnswer& item : answers) {
  std::cout << absl::StrFormat("Text: %s logit=%f start=%d end=%d", item.text,
                               item.pos.logit, item.pos.start, item.pos.end)
            << std::endl;
}
// Output:
// Text: 10 July 1856 logit=16.8527 start=17 end=19
// ... (and more)
//
// So the top-1 answer is: "10 July 1856".
```

In the above code, `item.text` is the text content of an answer. We use a span
with closed interval `[item.pos.start, item.pos.end]` to denote predicted tokens
in the answer, and `item.pos.logit` is the sum of span logits to represent the
confidence score.

### NLClassifier

`NLClassifier` API is able to load any TFLite models for natural language
classaification task such as language detection or sentiment detection.

The API expects a TFLite model with the following input/output tensor:
Input tensor0:
  (kTfLiteString) - input of the model, accepts a string.
Output tensor0:
  (kTfLiteUInt8/kTfLiteInt8/kTfLiteInt16/kTfLiteFloat32/kTfLiteFloat64)
  - output scores for each class, if type is one of the Int types,
    dequantize it to double
Output tensor1: optional
  (kTfLiteString)
  - output classname for each class, should be of the same length with
    scores. If this tensor is not present, the API uses score indices as
    classnames.
By default the API tries to find the input/output tensors with default
configurations in NLClassifierOptions, with tensor name prioritized over
tensor index. The option is configurable for different TFLite models.

Use the C++ API to perform language ID classification as follows:

```cc
using tflite::support::task::text::nlclassifier::NLClassifier;
using tflite::support::task::core::Category;
auto classifier = NLClassifier::CreateNLClassifier("/path/to/model");
// Or create a customized NLClassifierOptions
// NLClassifierOptions options =
//   {
//     .output_score_tensor_name = myOutputScoreTensorName,
//     .output_label_tensor_name = myOutputLabelTensorName,
//   }
// auto classifier = NLClassifier::CreateNLClassifier("/path/to/model", options);
std::string context = "What language is this?";
std::vector<Category> categories = classifier->Classify(context);
// Access category results.
for (const Categoryr& category : categories) {
  std::cout << absl::StrFormat("Language: %s Probability: %f", category.class_name, category_.score)
            << std::endl;
}
// Output:
// Language: en Probability=0.9
// ... (and more)
//
// So the top-1 answer is 'en'.
```
