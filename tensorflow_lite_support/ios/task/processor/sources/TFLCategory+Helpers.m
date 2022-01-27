/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#import "tensorflow_lite_support/ios/task/processor/sources/TFLCategory+Helpers.h"

@implementation TFLCategory (Helpers)

+ (TFLCategory *)categoryWithCCategory:(TfLiteCategory *)cCategory {
  if (cCategory == nil) return nil;

  TFLCategory *category = [[TFLCategory alloc] init];

  if (cCategory->display_name != nil) {
    category.displayName = [NSString stringWithCString:cCategory->display_name
                                              encoding:NSUTF8StringEncoding];
  }

  if (cCategory->label != nil) {
    category.label = [NSString stringWithCString:cCategory->label encoding:NSUTF8StringEncoding];
  }

  category.score = cCategory->score;
  category.classIndex = (NSInteger)cCategory->index;

  return category;
}
@end
