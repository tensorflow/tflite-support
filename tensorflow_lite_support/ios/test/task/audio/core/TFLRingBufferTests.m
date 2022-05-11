/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#import <XCTest/XCTest.h>

#import "tensorflow_lite_support/ios/task/audio/core/sources/TFLRingBuffer.h"
#import "tensorflow_lite_support/ios/sources/TFLCommon.h"

#define VerifyError(error, expectedErrorDomain, expectedErrorCode, expectedLocalizedDescription) \
  XCTAssertEqual(error.domain, expectedErrorDomain);                                             \
  XCTAssertEqual(error.code, expectedErrorCode);                                             \
  XCTAssertEqualObjects(error.localizedDescription, expectedLocalizedDescription);                                             \

NS_ASSUME_NONNULL_BEGIN

@interface TFLRingBufferTests : XCTestCase
@end

@implementation TFLRingBufferTests

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the
  // class.
  [super setUp];
}

-(void)testSuccessfulLoadFullLength {

   NSInteger inDataLength = 5;
   float inData[] = {1, 2, 3, 4, 5};

   TFLFloatBuffer *inBuffer = [[TFLFloatBuffer alloc] initWithData:&(inData[0]) size:inDataLength];
    
   NSInteger bufferSize = 5;
   TFLRingBuffer *ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:bufferSize];

   XCTAssertTrue([ringBuffer loadBuffer:inBuffer offset:0 size:inDataLength error:nil]);

   TFLFloatBuffer *outBuffer = [ringBuffer floatBuffer];
    
   XCTAssertEqual(outBuffer.size, inDataLength);
     
   float expectedData[] = {1, 2, 3, 4, 5};

   for (int i = 0; i < inDataLength; i++) {
     XCTAssertEqual(outBuffer.data[i], inData[i]);
    }  
}

-(void)testSuccessfulLoadPartialBuffer {

   NSInteger inDataSize = 3;
   float inData[] = {1.0f, 2.0f, 3.0f};
   TFLFloatBuffer *inBuffer = [[TFLFloatBuffer alloc] initWithData:&(inData[0]) size:inDataSize];
    
   NSInteger bufferSize = 5;
   TFLRingBuffer *ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:bufferSize];

   XCTAssertTrue([ringBuffer loadBuffer:inBuffer offset:0 size:inDataSize error:nil]);

   TFLFloatBuffer *outBuffer = [ringBuffer floatBuffer];
    
   XCTAssertEqual(outBuffer.size, bufferSize);
   
   float expectedData[] = {0.0f, 0.0f, 1.0f, 2.0f, 3.0f};

   for (int i = 0; i < bufferSize; i++) {
      XCTAssertEqual(outBuffer.data[i], expectedData[i]);
    }  
}

-(void)testSuccessfulLoadByShiftingOutOldElements {

   NSInteger initialDataSize = 4;
  float initialArray[] = {1.0f, 2.0f, 3.0f, 4.0f};
   
   TFLFloatBuffer *initialBuffer = [[TFLFloatBuffer alloc] initWithData:&(initialArray[0]) size:initialDataSize];
    
  NSInteger bufferSize = 5;
   TFLRingBuffer *ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:bufferSize];

   XCTAssertTrue([ringBuffer loadBuffer:initialBuffer offset:0 size:initialDataSize error:nil]);

   NSInteger inDataSize = 3;
   float inArray[] = {5, 6, 7};
   TFLFloatBuffer *inBuffer = [[TFLFloatBuffer alloc] initWithData:&(inArray[0]) size:inDataSize];

   XCTAssertTrue([ringBuffer loadBuffer:inBuffer offset:0 size:inDataSize error:nil]);

   TFLFloatBuffer *outBuffer = [ringBuffer floatBuffer];
    
   XCTAssertEqual(outBuffer.size, bufferSize);
   
   float expectedData[] = {3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
   
   for (int i = 0; i < bufferSize; i++) {
       XCTAssertEqual(outBuffer.data[i], expectedData[i]);
     }
}

-(void)testSuccessfulLoadWithMostRecentElements {

   NSInteger initialDataSize = 5;
  float initialArray[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
   
   TFLFloatBuffer *initialBuffer = [[TFLFloatBuffer alloc] initWithData:&(initialArray[0]) size:initialDataSize];
    
   NSInteger bufferSize = 5;
   TFLRingBuffer *ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:5];

   XCTAssertTrue([ringBuffer loadBuffer:initialBuffer offset:0 size:initialDataSize error:nil]);

   NSInteger inDataSize = 6;
   float inArray[] = {6, 7, 8, 9, 10, 11};
   TFLFloatBuffer *inBuffer = [[TFLFloatBuffer alloc] initWithData:&(inArray[0]) size:inDataSize];

   XCTAssertTrue([ringBuffer loadBuffer:inBuffer offset:0 size:inDataSize error:nil]);

   TFLFloatBuffer *outBuffer = [ringBuffer floatBuffer];
    
   XCTAssertEqual(outBuffer.size, bufferSize);
  
   float expectedData[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

   for (int i = 0; i < bufferSize; i++) {
      XCTAssertEqual(outBuffer.data[i], expectedData[i]);
    }  
}

-(void)testSuccessfulLoadWithOffseAndMostRecentElements{

   NSInteger initialDataSize = 5;
  float initialArray[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
   
   TFLFloatBuffer *initialBuffer = [[TFLFloatBuffer alloc] initWithData:&(initialArray[0]) size:initialDataSize];
    
   NSInteger bufferSize = 5;
   TFLRingBuffer *ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:5];

   XCTAssertTrue([ringBuffer loadBuffer:initialBuffer offset:0 size:initialDataSize error:nil]);

   NSInteger totalInSize = 8;
   float inArray[] = {6, 7, 8, 9, 10, 11, 12, 13};
   TFLFloatBuffer *inBuffer = [[TFLFloatBuffer alloc] initWithData:&(inArray[0]) size:totalInSize];
   
   NSInteger offset = 2;
   NSInteger inDataSize = 6;
   XCTAssertTrue([ringBuffer loadBuffer:inBuffer offset:offset size:inDataSize error:nil]);

   TFLFloatBuffer *outBuffer = [ringBuffer floatBuffer];
    
   XCTAssertEqual(outBuffer.size, bufferSize);
   
   float expectedData[] = {9.0f, 10.0f, 11.0f, 12.0f, 13.0f};

   for (int i = 0; i < bufferSize; i++) {
      XCTAssertEqual(outBuffer.data[i], expectedData[i]);
    }  
}

-(void)testLoadSucceedsWithOffset {

   NSInteger initialDataSize = 2;
  float initialArray[] = {1.0f, 2.0f};
   
   TFLFloatBuffer *initialBuffer = [[TFLFloatBuffer alloc] initWithData:&(initialArray[0]) size:initialDataSize];
    
   NSInteger bufferSize = 5;
   TFLRingBuffer *ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:5];

   XCTAssertTrue([ringBuffer loadBuffer:initialBuffer offset:0 size:initialDataSize error:nil]);

   NSInteger totalInSize = 4;
   float inArray[] = {6.0f, 7.0f, 8.0f, 9.0f};
   TFLFloatBuffer *inBuffer = [[TFLFloatBuffer alloc] initWithData:&(inArray[0]) size:totalInSize];
   
   NSInteger offset = 2;
   NSInteger inDataSize = 2;
   XCTAssertTrue([ringBuffer loadBuffer:inBuffer offset:offset size:inDataSize error:nil]);

   TFLFloatBuffer *outBuffer = [ringBuffer floatBuffer];
    
   XCTAssertEqual(outBuffer.size, bufferSize);
   
   float expectedData[] = {0.0f, 1.0f, 2.0f, 8.0f, 9.0f};
   
   for (int i = 0; i < bufferSize; i++) {
        XCTAssertEqual(outBuffer.data[i], expectedData[i]);
   }   
}

-(void)testLoadFailsWithIndexOutofBounds {

   NSInteger initialDataSize = 2;
  float initialArray[] = {1.0f, 2.0f};
   
   TFLFloatBuffer *initialBuffer = [[TFLFloatBuffer alloc] initWithData:&(initialArray[0]) size:initialDataSize];
    
   NSInteger bufferSize = 5;
   TFLRingBuffer *ringBuffer = [[TFLRingBuffer alloc] initWithBufferSize:5];

   XCTAssertTrue([ringBuffer loadBuffer:initialBuffer offset:0 size:initialDataSize error:nil]);

   NSInteger totalInSize = 4;
   float inArray[] = {6.0f, 7.0f, 8.0f, 9.0f};
   TFLFloatBuffer *inBuffer = [[TFLFloatBuffer alloc] initWithData:&(inArray[0]) size:totalInSize];
   
   NSInteger offset = 2;
   NSInteger inDataSize = 3;

   NSError *error = nil;
   XCTAssertFalse([ringBuffer loadBuffer:inBuffer offset:offset size:inDataSize error:&error]);

   XCTAssertNotNil(error);
   VerifyError(error,@"org.tensorflow.lite.tasks",TFLSupportErrorCodeInvalidArgumentError,@"offset + size exceeds the maximum size of the source buffer.");

}

@end

NS_ASSUME_NONNULL_END
