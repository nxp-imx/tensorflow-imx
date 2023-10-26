load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_jni_library")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

# Following targets are using for testing selective-built flex delegate
# in Java. Please don't use them for other purposes.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/lite/android:__subpackages__",
    ],
    licenses = ["notice"],
)

tflite_flex_jni_library(
    name = "test",
    testonly = 1,
    additional_deps = ["//tensorflow/lite/python/testdata:double_op_and_kernels"],
    models = [
        "//tensorflow/lite:testdata/multi_add_flex.bin",
        "//tensorflow/lite:testdata/double_flex.bin",
    ],
)

java_library(
    name = "test_tensorflowlitelib_flex",
    testonly = 1,
    srcs = ["//tensorflow/lite/delegates/flex/java/src/main/java/org/tensorflow/lite/flex:flex_delegate"],
    javacopts = JAVACOPTS,
    visibility = ["//visibility:private"],
    deps = [
        ":libtensorflowlite_flex_jni.so",
        "//tensorflow/lite/java:tensorflowlitelib",
        "@org_checkerframework_qual",
    ],
)

java_test(
    name = "SelectiveBuiltInterpreterFlexTest",
    size = "small",
    srcs = [
        "//tensorflow/lite/java:portable_flex_tests",
        "//tensorflow/lite/java:portable_test_utils",
    ],
    data = [
        "//tensorflow/lite:testdata/multi_add_flex.bin",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_cuda_on_cpu_tap",  # CUDA + flex is not officially supported.
        "no_gpu",  # GPU + flex is not officially supported.
        "no_oss",  # Currently requires --config=monolithic, b/118895218.
    ],
    test_class = "org.tensorflow.lite.InterpreterFlexTest",
    visibility = ["//visibility:private"],
    deps = [
        ":test_tensorflowlitelib_flex",
        "//tensorflow/lite/java:tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "SelectiveBuiltInterpreterFlexWithCustomOpsTest",
    size = "small",
    srcs = [
        "//tensorflow/lite/java:portable_flex_with_custom_ops_tests",
        "//tensorflow/lite/java:portable_test_utils",
    ],
    data = [
        "//tensorflow/lite:testdata/double_flex.bin",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_cuda_on_cpu_tap",  # CUDA + flex is not officially supported.
        "no_gpu",  # GPU + flex is not officially supported.
        "no_oss",  # Currently requires --config=monolithic, b/118895218.
    ],
    test_class = "org.tensorflow.lite.InterpreterFlexWithCustomOpsTest",
    visibility = ["//visibility:private"],
    deps = [
        ":test_tensorflowlitelib_flex",
        "//tensorflow/lite/java:tensorflowlitelib",
        "@com_google_truth",
        "@junit",
    ],
)

# For build test only.
ios_static_framework(
    name = "TestTensorFlowLiteSelectTfOps_framework",
    testonly = 1,
    avoid_deps = ["//tensorflow/lite/core/c:common"],
    bundle_name = "TestTensorFlowLiteSelectTfOps",
    minimum_os_version = "12.0",
    deps = [
        ":test_flex_delegate",
    ],
)

build_test(
    name = "framework_build_test",
    # build_test targets are not meant to be run with sanitizers, and can also
    # cause problems with coverage testing.
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "nozapfhahn",
    ],
    targets = [
        ":TestTensorFlowLiteSelectTfOps_framework",
    ],
)

# Embedded Linux: Flex delegate testing using benchmark_model
exports_files(["simple_flex_model_int8.tflite"])

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_shared_library", "tflite_flex_cc_library")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")

# Embedded Linux: Reduced size, static
tflite_flex_cc_library(
    name = "delegate_reduced",
    models = [
        "simple_flex_model_int8.tflite",
    ],
    visibility = ["//visibility:public"],
)

tf_cc_binary(
    name = "benchmark_model_plus_flex_reduced",
    srcs = [
        "//tensorflow/lite/tools/benchmark:benchmark_plus_flex_main.cc",
    ],
    copts = tflite_copts() + tflite_copts_warnings(),
    linkopts = tflite_linkopts(),
    deps = [
        ":delegate_reduced",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
        "//tensorflow/lite/testing:init_tensorflow",
        "//tensorflow/lite/tools:logging",
    ],
)

# Embedded Linux: Reduced size, dynamically linked 
tflite_flex_shared_library(
  name = "tensorflowlite_flex_reduced",
  models = [
    "simple_flex_model_int8.tflite",
  ],
  visibility = ["//visibility:public"]
)

cc_import(
    name = "libtensorflowlite_flex_reduced",
    shared_library = ":tensorflowlite_flex_reduced",
)

tf_cc_binary(
    name = "benchmark_model_plus_flex_dynamic_reduced",
    srcs = [
        "//tensorflow/lite/tools/benchmark:benchmark_plus_flex_main.cc",
    ],
    copts = tflite_copts() + tflite_copts_warnings(),
    linkopts = tflite_linkopts(),
    deps = [
        ":libtensorflowlite_flex_reduced",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
        "//tensorflow/lite/testing:init_tensorflow",
        "//tensorflow/lite/tools:logging",
    ],
)

# Embedded Linux: Full size, static
tflite_flex_cc_library(
    name = "delegate_full",
    visibility = ["//visibility:public"],
)

tf_cc_binary(
    name = "benchmark_model_plus_flex_full",
    srcs = [
        "//tensorflow/lite/tools/benchmark:benchmark_plus_flex_main.cc",
    ],
    copts = tflite_copts() + tflite_copts_warnings(),
    linkopts = tflite_linkopts(),
    deps = [
        ":delegate_full",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
        "//tensorflow/lite/testing:init_tensorflow",
        "//tensorflow/lite/tools:logging",
    ],
)

# Embedded Linux: Full size, dynamically linked 
tflite_flex_shared_library(
  name = "tensorflowlite_flex_full",
  visibility = ["//visibility:public"]
)

cc_import(
    name = "libtensorflowlite_flex_full",
    shared_library = ":tensorflowlite_flex_full",
)

tf_cc_binary(
    name = "benchmark_model_plus_flex_dynamic_full",
    srcs = [
        "//tensorflow/lite/tools/benchmark:benchmark_plus_flex_main.cc",
    ],
    copts = tflite_copts() + tflite_copts_warnings(),
    linkopts = tflite_linkopts(),
    deps = [
        ":libtensorflowlite_flex_full",
        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",
        "//tensorflow/lite/testing:init_tensorflow",
        "//tensorflow/lite/tools:logging",
    ],
)