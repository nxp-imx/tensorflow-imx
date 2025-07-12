/*
* Copyright 2025 NXP
*
* SPDX-License-Identifier: Apache-2.0
*
*/
#include <jni.h>
#include <sstream>
#include "tensorflow/lite/delegates/external/external_delegate.h"

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_external_ExternalDelegate_createDelegate(
    JNIEnv* env, jclass clazz, jstring lib_path, jobjectArray keys, jobjectArray values) {
  // Example: set the path to your external delegate shared library
  const char* clib_path = env->GetStringUTFChars(lib_path, nullptr);
  if (!clib_path) {
    // You may want to throw an exception or handle this error
    return 0;
  }
  TfLiteExternalDelegateOptions options = TfLiteExternalDelegateOptionsDefault(clib_path);
  // Optionally, insert more options here using TfLiteExternalDelegateOptionsInsert
  TfLiteDelegate* delegate = TfLiteExternalDelegateCreate(&options);
  env->ReleaseStringUTFChars(lib_path, clib_path);

  jsize length = env->GetArrayLength(keys);
  if (length != env->GetArrayLength(values)) {
    return 0;
  }

  for (jsize i = 0; i < length; ++i) {
      jstring key = static_cast<jstring>(env->GetObjectArrayElement(keys, i));
      jstring value = static_cast<jstring>(env->GetObjectArrayElement(values, i));
      const char* ckey = env->GetStringUTFChars(key, nullptr);
      const char* cvalue = env->GetStringUTFChars(value, nullptr);
      options.insert(&options, ckey, cvalue);
      env->ReleaseStringUTFChars(key, ckey);
      env->ReleaseStringUTFChars(value, cvalue);
  }

  return reinterpret_cast<jlong>(delegate);
}

JNIEXPORT void JNICALL
Java_org_tensorflow_lite_external_ExternalDelegate_deleteDelegate(
    JNIEnv* env, jclass clazz, jlong delegate) {
  TfLiteExternalDelegateDelete(reinterpret_cast<TfLiteDelegate*>(delegate));
}

}  // extern "C"
