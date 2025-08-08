#
# Copyright 2025 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary", "tflite_jni_linkopts")
load("//tensorflow/lite/core/shims:cc_library_with_tflite.bzl", "android_library_with_tflite", "cc_library_with_tflite", "java_library_with_tflite", "java_test_with_tflite", "jni_binary_with_tflite")


package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# EXPERIMENTAL: Native target that runs inference on the Hexagon backend.
# The Hexagon backend-related targets are intentionally not included in BUILD.bazel.
jni_binary_with_tflite(
    name = "libtensorflowlite_external_delegate_jni.so",
    srcs = [ "src/main/native/external_delegate_jni.cc" ],
    deps = [
        "//tensorflow/lite/java/jni",
        "//tensorflow/lite/delegates/external:external_delegate",
    ],
)

cc_library(
    name = "tensorflowlite_external_delegate",
    srcs = [
        "libtensorflowlite_external_delegate_jni.so",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "external_delegate_group",
    srcs = [
        "src/main/java/org/tensorflow/lite/external/ExternalDelegate.java"
    ],
    visibility = ["//visibility:public"],
)

android_library(
    name = "external_delegate",
    srcs = ["src/main/java/org/tensorflow/lite/external/ExternalDelegate.java"],
    deps = [
        "//tensorflow/lite/java:tensorflowlite_java",
    ],
)
