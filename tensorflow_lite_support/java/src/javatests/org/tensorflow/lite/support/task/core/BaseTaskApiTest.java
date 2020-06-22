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

package org.tensorflow.lite.support.task.core;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import androidx.test.core.app.ApplicationProvider;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.robolectric.RobolectricTestRunner;

/** Test for {@link BaseTaskApi}. */
@RunWith(RobolectricTestRunner.class)
public class BaseTaskApiTest {
  private static final String MODEL_FILE1 = "file1";
  private static final String MODEL_FILE2 = "file2";
  private static final String NON_EXISTENT_FILE = "i_do_not_exist";

  @Test
  public void initializeWithoutFile() {
    TestTaskApi api = TestTaskApi.createTestTaskApi();

    assertNotNull(api);
  }

  @Test
  public void initializeWithFiles() {
    TestTaskApi api =
        TestTaskApi.createTestTaskApiWithBuffers(
            ApplicationProvider.getApplicationContext(), MODEL_FILE1, MODEL_FILE2);

    assertNotNull(api);
  }

  @Test
  public void initializeWithNonExistentFiles_fail() {
    assertThrows(
        IllegalStateException.class,
        () ->
            TestTaskApi.createTestTaskApiWithBuffers(
                ApplicationProvider.getApplicationContext(), MODEL_FILE1, NON_EXISTENT_FILE));
  }

  @Test
  public void initialize_invokeAdd() {
    TestTaskApi api = spy(TestTaskApi.createTestTaskApi());

    int i1 = 1;
    int i2 = 2;

    assertEquals(i1 + i2, api.add(i1, i2));
    verify(api).addNative(eq(api.getNativeHandle()), eq(i1), eq(i2));
  }

  @Test
  public void initialize_close_once_verify_lib_removed_only_once() {
    TestTaskApi api = spy(TestTaskApi.createTestTaskApi());

    long nativeHandle = api.getNativeHandle();

    assertNotEquals(nativeHandle, 0L);

    api.close();
    api.close();
    api.close();

    verify(api, times(1)).deinitJni(eq(nativeHandle));
    assertTrue(api.isClosed());
  }
}
