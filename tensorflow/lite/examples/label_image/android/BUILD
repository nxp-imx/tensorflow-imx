#
# Copyright 2025 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

load("//tensorflow/lite:build_def.bzl", "tflite_jni_binary")
load("@build_bazel_rules_android//android:rules.bzl", "android_binary")
load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/benchmark/android:build_defs.bzl", "get_shared_libs")
#load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")

SHARED_LIBS, SHARED_LIBS_EXTRA_DEPS = get_shared_libs()

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
    ] + SHARED_LIBS_EXTRA_DEPS
)

[cc_import(
    name = "%s" % lib_name,
    shared_library = "%s" % lib_path,
) for lib_name, lib_path in SHARED_LIBS]

