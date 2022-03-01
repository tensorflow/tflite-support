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
#import "tensorflow_lite_support/ios/task/processor/sources/TFLSegmentationResult.h"

@implementation TFLColoredLabel
@synthesize r;
@synthesize g;
@synthesize b;
@synthesize label;
@synthesize displayName;
@end

@implementation TFLSegmentation
@synthesize width;
@synthesize height;
@synthesize confidenceMasks;
@synthesize categoryMask;
@synthesize coloredLabelsSize;
@synthesize coloredLabels;

- (void)dealloc {
  if (self.confidenceMasks != NULL) {
    for (int j = 0; j < self.coloredLabelsSize; ++j) {
      free(self.confidenceMasks[j]);
    }
    free(self.confidenceMasks);
  } else if (self.categoryMask != NULL) {
    free(self.categoryMask);
  }
}

@end

@implementation TFLSegmentationResult
@synthesize segmentations;

@end
