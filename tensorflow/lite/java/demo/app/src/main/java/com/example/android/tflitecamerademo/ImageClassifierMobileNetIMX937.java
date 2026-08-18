/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package com.example.android.tflitecamerademo;

import android.app.Activity;
import java.io.IOException;

/**
 * This classifier works with the Neutron MobileNet model.
 */
public class ImageClassifierMobileNetIMX937 extends ImageClassifierINT8MobileNet {

  /**
   * Initializes an {@code ImageClassifierINT8MobileNet}.
   *
   * @param activity
   */
  ImageClassifierMobileNetIMX937(Activity activity) throws IOException {
    super(activity);
  }

  @Override
  protected String getModelPath() {
    return "mobilenet_v1_tf1.x_int8_imx937.tflite";
  }
}
