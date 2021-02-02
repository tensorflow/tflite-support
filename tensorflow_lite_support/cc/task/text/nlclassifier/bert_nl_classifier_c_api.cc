/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_lite_support/cc/task/text/nlclassifier/bert_nl_classifier_c_api.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "tensorflow_lite_support/cc/task/core/category.h"
#include "tensorflow_lite_support/cc/task/text/nlclassifier/bert_nl_classifier.h"
#include "tensorflow_lite_support/cc/task/text/proto/bert_nl_classifier_options_proto_inc.h"

using CategoryCPP = ::tflite::task::core::Category;
using BertNLClassifierCPP =
    ::tflite::task::text::nlclassifier::BertNLClassifier;
using BertNLClassifierOptionsCPP =
    ::tflite::task::text::BertNLClassifierOptions;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct BertNLClassifier {
  std::unique_ptr<BertNLClassifierCPP> impl;
};

const struct BertNLClassifierOptions BertNLClassifierOptions_default = {128};

BertNLClassifier* BertNLClassifierFromFileAndOptions(
    const char* model_path, const struct BertNLClassifierOptions* options) {
  BertNLClassifierOptionsCPP cc_options;
  cc_options.set_max_seq_len(options->max_seq_len);
  cc_options.mutable_base_options()->mutable_model_file()->set_file_name(
      model_path);
  auto classifier_status = BertNLClassifierCPP::CreateFromOptions(cc_options);
  if (classifier_status.ok()) {
    return new BertNLClassifier{.impl = std::unique_ptr<BertNLClassifierCPP>(
                                    dynamic_cast<BertNLClassifierCPP*>(
                                        classifier_status.value().release()))};
  } else {
    return nullptr;
  }
}

BertNLClassifier* BertNLClassifierFromFile(const char* model_path) {
  return BertNLClassifierFromFileAndOptions(model_path,
                                            &BertNLClassifierOptions_default);
}

Categories* BertNLClassifierClassify(const BertNLClassifier* classifier,
                                     const char* text) {
  std::vector<CategoryCPP> results =
      classifier->impl->Classify(absl::string_view(text).data());
  size_t size = results.size();
  auto* categories = new Category[size];

  for (size_t i = 0; i < size; ++i) {
    categories[i].text = strdup(results[i].class_name.c_str());
    categories[i].score = results[i].score;
  }

  auto* c_categories = new Categories;
  c_categories->size = size;
  c_categories->categories = categories;
  return c_categories;
}

void BertNLClassifierDelete(BertNLClassifier* classifier) { delete classifier; }

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
