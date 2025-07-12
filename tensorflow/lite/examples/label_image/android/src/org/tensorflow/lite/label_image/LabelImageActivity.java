/*
* Copyright 2025 NXP
*
* SPDX-License-Identifier: Apache-2.0
*
*/

package org.tensorflow.lite.label_image;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Trace;
import android.util.Log;

import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/** Main {@code Activity} class for the label_image app. */
public class LabelImageActivity extends Activity {

  private static final String TAG = "LabelImageActivity";
  private static final String MODEL = "/data/local/tmp/mobilenet_v1_1.0_224_quant.tflite";
  private static final String LABEL = "/data/local/tmp/labels.txt";
  private static final String IMAGE = "/data/local/tmp/grace_hopper.bmp";

  @Override
  public void onCreate(Bundle savedInstanceState) {
    Log.i(TAG, "Label image application starts.");

    super.onCreate(savedInstanceState);

    String model, label, image, ext_delegate, ext_delegate_options;
    int threads;

    Intent intent = getIntent();
    Bundle bundle = intent.getExtras();
    if (bundle != null) {
        model = bundle.getString("graph", MODEL);
        label = bundle.getString("label", LABEL);
        image = bundle.getString("image", IMAGE);
        ext_delegate = bundle.getString("ext_delegate", null);
        ext_delegate_options = bundle.getString("ext_delegate_options", null);
        threads = bundle.getInt("num_threads", 1);
    } else {
        model = MODEL;
	label = LABEL;
	image = IMAGE;
	threads = 1;
	ext_delegate = null;
	ext_delegate_options = null;
    }


        Log.i(TAG, "Running TensorFlow Lite classification inference");
        LabelImage labelimage = new LabelImage(threads, ext_delegate, ext_delegate_options);
       // try {
            labelimage.loadModelAndLabel(model, label);
        //} catch (Exception e) {
         //   Log.e(TAG, "Fail to load model." + e);
        //}
        labelimage.inference(image);
        Trace.endSection();
    finish();
  }
}

