/*
* Copyright 2025 NXP
*
* SPDX-License-Identifier: Apache-2.0
*
*/
package org.tensorflow.lite.external;

import org.tensorflow.lite.Delegate;
import java.util.List;
import java.util.ArrayList;


public class ExternalDelegate implements Delegate {

  private static final long INVALID_DELEGATE_HANDLE = 0;

  private long delegateHandle;

  static {
    System.loadLibrary("tensorflowlite_external_delegate_jni");
  }

  public ExternalDelegate(Options options) {
    String[] jStrKeys = options.getKeys().toArray(new String[0]);
    String[] jStrValues = options.getValues().toArray(new String[0]);
    delegateHandle =
        createDelegate(
            options.getLibPath(),
            jStrKeys,
            jStrValues
        );
    if (delegateHandle == INVALID_DELEGATE_HANDLE) {
      throw new UnsupportedOperationException(
          "This Device doesn't support External Delegate execution.");
    }
  }

  public long getNativeHandle() {
    return delegateHandle;
  }


  public void close() {
    if (delegateHandle != INVALID_DELEGATE_HANDLE) {
      deleteDelegate(delegateHandle);
      delegateHandle = INVALID_DELEGATE_HANDLE;
    }
  }

  private static native long createDelegate(
      String lib_path,
      String[] keys,
      String[] values
  );

  private static native void deleteDelegate(long delegateHandle);

  public static class Options {
    private String libPath;
    private List<String> keys = new ArrayList<>();
    private List<String> values = new ArrayList<>();

    public Options(String ext_delegate_path) {
      this.libPath = ext_delegate_path;
    }

    public Options(String ext_delegate_path, List<String> keys, List<String> values) {
      this.libPath = ext_delegate_path;
      this.keys = keys;
      this.values = values;
    }

    public void insert(String key, String value) {
      keys.add(key);
      values.add(value);
    }

    public String getLibPath() {
      return libPath;
    }

    public List<String> getKeys() {
      return keys;
    }

    public List<String> getValues() {
      return values;
    }
  }
}
