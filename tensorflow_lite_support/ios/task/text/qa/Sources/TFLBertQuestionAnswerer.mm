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
#import "third_party/tensorflow_lite_support/ios/task/text/qa/Sources/TFLBertQuestionAnswerer.h"
#import "third_party/objective_c/google_toolbox_for_mac/GTMDefines.h"
#import "third_party/tensorflow_lite_support/ios/utils/Sources/TFLStringUtil.h"

#include "third_party/tensorflow_lite_support/cc/task/text/qa/bert_question_answerer.h"

NS_ASSUME_NONNULL_BEGIN
using BertQuestionAnswererCPP = ::tflite::support::task::text::qa::BertQuestionAnswerer;
using QuestionAnswererCPP = ::tflite::support::task::text::qa::QuestionAnswerer;
using QaAnswerCPP = ::tflite::support::task::text::qa::QaAnswer;

@implementation TFLQAAnswer
@synthesize pos;
@synthesize text;
@end

@implementation TFLBertQuestionAnswerer {
  std::unique_ptr<QuestionAnswererCPP> _bertQuestionAnswerwer;
}

+ (instancetype)mobilebertQuestionAnswererWithModelPath:(NSString *)modelPath
                                              vocabPath:(NSString *)vocabPath {
  absl::StatusOr<std::unique_ptr<QuestionAnswererCPP>> cQuestionAnswerer =
      BertQuestionAnswererCPP::CreateBertQuestionAnswerer(MakeString(modelPath),
                                                          MakeString(vocabPath));
  _GTMDevAssert(cQuestionAnswerer.ok(), @"Failed to create BertQuestionAnswerer");
  return [[TFLBertQuestionAnswerer alloc]
      initWithQuestionAnswerer:std::move(cQuestionAnswerer.value())];
}

+ (instancetype)albertQuestionAnswererWithModelPath:(NSString *)modelPath
                              setencepieceModelPath:(NSString *)setencepieceModelPath {
  absl::StatusOr<std::unique_ptr<QuestionAnswererCPP>> cQuestionAnswerer =
      BertQuestionAnswererCPP::CreateAlbertQuestionAnswerer(MakeString(modelPath),
                                                            MakeString(setencepieceModelPath));
  _GTMDevAssert(cQuestionAnswerer.ok(), @"Failed to create BertQuestionAnswerer");
  return [[TFLBertQuestionAnswerer alloc]
      initWithQuestionAnswerer:std::move(cQuestionAnswerer.value())];
}

- (instancetype)initWithQuestionAnswerer:
    (std::unique_ptr<QuestionAnswererCPP>)bertQuestionAnswerer {
  self = [super init];
  if (self) {
    _bertQuestionAnswerwer = std::move(bertQuestionAnswerer);
  }
  return self;
}

- (NSMutableArray<TFLQAAnswer *> *)arrayFromVector:(std::vector<QaAnswerCPP>)vector {
  NSMutableArray<TFLQAAnswer *> *ret = [NSMutableArray arrayWithCapacity:vector.size()];

  for (int i = 0; i < vector.size(); i++) {
    QaAnswerCPP answerCpp = vector[i];
    TFLQAAnswer *answer = [[TFLQAAnswer alloc] init];
    [answer setPos:{.start = answerCpp.pos.start,
                    .end = answerCpp.pos.end,
                    .logit = answerCpp.pos.logit}];
    [answer setText:MakeNSString(answerCpp.text)];
    [ret addObject:answer];
  }
  return ret;
}

- (NSArray<TFLQAAnswer *> *)answerWithContext:(NSString *)context question:(NSString *)question {
  std::vector<QaAnswerCPP> results =
      _bertQuestionAnswerwer->Answer(MakeString(context), MakeString(question));
  return [self arrayFromVector:results];
}
@end
NS_ASSUME_NONNULL_END
