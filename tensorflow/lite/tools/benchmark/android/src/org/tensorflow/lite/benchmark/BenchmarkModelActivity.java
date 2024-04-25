/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.benchmark;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.os.Trace;
import android.util.Log;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.file.DirectoryStream;
import java.util.Iterator;
import java.io.IOException;

/** Main {@code Activity} class for the benchmark app. */
public class BenchmarkModelActivity extends Activity {

  private static final String TAG = "tflite_BenchmarkModelActivity";

  private static final String ARGS_INTENT_KEY_0 = "args";
  private static final String ARGS_INTENT_KEY_1 = "--args";

  @Override
  public void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    Intent intent = getIntent();
    Bundle bundle = intent.getExtras();
    String args = bundle.getString(ARGS_INTENT_KEY_0, bundle.getString(ARGS_INTENT_KEY_1));
    if (args.contains("--use_hexagon=true") || args.contains("--use_hexagon=1")) {
      // Users should not specify this argument.
      args = args + " --hexagon_lib_path=" + getApplicationInfo().nativeLibraryDir;
    }
    else if (args.contains("--use_external_delegate=true") || args.contains("--use_external_delegate=1")) {
      final Path dir = Paths.get(getApplicationInfo().nativeLibraryDir);
      // Search for all delegate libraries in the application native library folder.
      try (DirectoryStream<Path> dirStream = Files.newDirectoryStream(dir, "lib*_delegate.so")) {
        Iterator<Path> itStream = dirStream.iterator();
        // If the list of libraries is not empty, apply the first delegate libary from it. 
        if (itStream.hasNext()) {
          // Users should not specify this argument.
          args = args.replaceFirst("--use_external_delegate=(true|1)", "--external_delegate_path=" + itStream.next().toString()); 
        }
        else {
          Log.e(TAG, "There is no delegate library in " + dir.toString());
          return;
        }
      } catch (IOException exception) {
        System.err.println("Failed to open external delegate library");
        System.exit(0);
      }
    }
    Log.i(TAG, "Running TensorFlow Lite benchmark with args: " + args);

    Trace.beginSection("TFLite Benchmark Model");
    BenchmarkModel.run(args);
    Trace.endSection();

    finish();
  }
}
