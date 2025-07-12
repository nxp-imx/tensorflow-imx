#
# Copyright 2025 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary", "tflite_jni_linkopts")
load("//tensorflow/lite/java:aar_with_jni.bzl", "aar_with_jni")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# EXPERIMENTAL: Native target that runs inference on the Hexagon backend.
# The Hexagon backend-related targets are intentionally not included in BUILD.bazel.
tflite_jni_binary(
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

android_library(
    name = "external_delegate",
    srcs = ["src/main/java/org/tensorflow/lite/external/ExternalDelegate.java"],
    deps = [
        "//tensorflow/lite/java:tensorflowlite_java",
    ],
)
