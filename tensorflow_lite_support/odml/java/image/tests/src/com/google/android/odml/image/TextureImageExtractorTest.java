/* Copyright 2021 Google LLC. All Rights Reserved.

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

package com.google.android.odml.image;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import android.opengl.EGLContext;
import java.nio.ByteBuffer;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.Mock;
import org.mockito.junit.MockitoJUnit;
import org.mockito.junit.MockitoRule;
import org.robolectric.RobolectricTestRunner;

/** Unit test for {@link TextureImageExtractor}. */
@RunWith(RobolectricTestRunner.class)
public class TextureImageExtractorTest {
  public static final int TEXTURE_NAME = 2;
  public static final int NATIVE_CONTEXT = 5;
  public static final int IMAGE_HEIGHT = 200;
  public static final int IMAGE_WIDTH = 100;
  public static final int IMAGE_FORMAT = MlImage.IMAGE_FORMAT_RGBA;

  @Rule public MockitoRule mockitoRule = MockitoJUnit.rule();
  @Mock private EGLContext eglContext;

  @Test
  public void extract_fromTextureImage_succeeds() {
    MlImage image =
        new TextureMlImageBuilder(
                TEXTURE_NAME, eglContext, IMAGE_WIDTH, IMAGE_HEIGHT, MlImage.IMAGE_FORMAT_RGBA)
            .setNativeContext(NATIVE_CONTEXT)
            .build();

    TextureFrame textureFrame = TextureImageExtractor.extract(image);

    assertThat(textureFrame.getTextureName()).isEqualTo(TEXTURE_NAME);
    assertThat(textureFrame.getEglContext()).isSameInstanceAs(eglContext);
    assertThat(textureFrame.getNativeContext()).isEqualTo(NATIVE_CONTEXT);
  }

  @Test
  public void extract_fromByteBuffer_throwsException() {
    ByteBuffer buffer = TestImageCreator.createRgbBuffer();
    MlImage image =
        new ByteBufferMlImageBuilder(
                buffer,
                TestImageCreator.getWidth(),
                TestImageCreator.getHeight(),
                MlImage.IMAGE_FORMAT_RGB)
            .build();

    assertThrows(IllegalArgumentException.class, () -> BitmapExtractor.extract(image));
  }
}
