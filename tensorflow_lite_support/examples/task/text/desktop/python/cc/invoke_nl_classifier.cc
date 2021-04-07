#include <iostream>
#include <limits>
#include <string.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/category.h"
#include "tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h"

namespace tflite {
namespace task {
namespace text {
namespace nlclassifier {

std::unique_ptr<NLClassifier> classifier;

void InitializeModel(int argc, char** argv){
  // Initialization
  classifier = NLClassifier::CreateFromFileAndOptions(
      argv[0],//model_path,
      {
        .input_tensor_name=argv[1],//kInputTensorName,
        .output_score_tensor_name=argv[2],//kOutputScoreTensorName,
      }).value();
}


void RunInference(int argc, char** argv, char **strings){
  // // Run inference
  std::vector<core::Category> categories = classifier->Classify(argv[0]);//kinput
  for (int i = 0; i < categories.size(); ++i) {
    const core::Category& category = categories[i];
    strcpy(strings[(i*2)], (category.class_name).c_str());
    strcpy(strings[(i*2)+1],std::to_string(category.score).c_str());
  }
}

}  // namespace nlclassifier
}  // namespace text
}  // namespace task
}  // namespace tflite

extern "C" {
  void InvokeInitializeModel(int argc, char** argv);
  void InvokeRunInference(int argc, char** argv, char **strings);
}

void InvokeInitializeModel(int argc, char** argv){
  tflite::task::text::nlclassifier::InitializeModel(argc, argv);
}

void InvokeRunInference(int argc, char** argv, char **strings){
  tflite::task::text::nlclassifier::RunInference(argc, argv, strings);
}
