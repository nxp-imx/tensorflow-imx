#
# Copyright 2025 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("@build_bazel_rules_android//android:rules.bzl", "android_binary")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
#load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")


android_binary(
    name = "label_image",
    srcs = glob(["src/**/*.java"]),
    custom_package = "org.tensorflow.lite.label_image",
    manifest = "AndroidManifest.xml",
    tags = ["manual"],
    deps = [
        "//tensorflow/lite/java:tensorflowlite",
	"//tensorflow/lite/delegates/external/java:tensorflowlite_external_delegate",
	"//tensorflow/lite/delegates/external/java:external_delegate",
    ]
)
