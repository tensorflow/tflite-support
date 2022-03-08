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
#import "tensorflow_lite_support/ios/task/processor/sources/TFLSegmentationResult+Helpers.h"

@implementation TFLSegmentationResult (Helpers)

+ (TFLSegmentationResult *)segmentationResultWithCResult:
    (TfLiteSegmentationResult *)cSegmentationResult {
  if (cSegmentationResult == nil) return nil;

  NSMutableArray *segmentations = [[NSMutableArray alloc] init];
  for (int i = 0; i < cSegmentationResult->size; i++) {
    TfLiteSegmentation cSegmentation = cSegmentationResult->segmentations[i];
    NSMutableArray *coloredLabels = [[NSMutableArray alloc] init];
    for (int j = 0; j < cSegmentation.colored_labels_size; j++) {
      TfLiteColoredLabel cColoredLabel = cSegmentation.colored_labels[j];

      TFLColoredLabel *coloredLabel = [[TFLColoredLabel alloc] init];
      coloredLabel.r = (NSUInteger)cColoredLabel.r;
      coloredLabel.g = (NSUInteger)cColoredLabel.g;
      coloredLabel.b = (NSUInteger)cColoredLabel.b;

      if (cColoredLabel.display_name != nil) {
        coloredLabel.displayName = [NSString stringWithCString:cColoredLabel.display_name
                                                      encoding:NSUTF8StringEncoding];
      }

      if (cColoredLabel.label != nil) {
        coloredLabel.label = [NSString stringWithCString:cColoredLabel.label
                                                encoding:NSUTF8StringEncoding];
      }

      [coloredLabels addObject:coloredLabel];
    }

    TFLSegmentation *segmentation = [[TFLSegmentation alloc] init];
    segmentation.coloredLabels = coloredLabels;

    NSInteger imageSize = cSegmentation.width * cSegmentation.height;

    if (cSegmentation.confidence_masks) {
      NSMutableArray *confidenceMasks = [[NSMutableArray alloc] init];
      for (int i = 0; i < cSegmentation.colored_labels_size; i++) {
        TFLConfidenceMask *confidenceMask = [[TFLConfidenceMask alloc] init];
        confidenceMask.width = (NSInteger)cSegmentation.width;
        confidenceMask.height = (NSInteger)cSegmentation.height;
        confidenceMask.mask = malloc(imageSize * sizeof(float));
        memcpy(confidenceMask.mask, cSegmentation.confidence_masks[i], imageSize * sizeof(float));
        [confidenceMasks addObject:confidenceMask];
      }
      segmentation.confidenceMasks = confidenceMasks;

    } else if (cSegmentation.category_mask) {
      TFLCategoryMask *categoryMask = [[TFLCategoryMask alloc] init];
      categoryMask.width = (NSInteger)cSegmentation.width;
      categoryMask.height = (NSInteger)cSegmentation.height;
      categoryMask.mask = malloc(imageSize * sizeof(UInt8));
      memcpy(categoryMask.mask, cSegmentation.category_mask,imageSize * sizeof(UInt8));
      segmentation.categoryMask = categoryMask;
    }

    [segmentations addObject:segmentation];
  }

  TFLSegmentationResult *segmentationResult = [[TFLSegmentationResult alloc] init];
  segmentationResult.segmentations = segmentations;
  return segmentationResult;
}
@end
