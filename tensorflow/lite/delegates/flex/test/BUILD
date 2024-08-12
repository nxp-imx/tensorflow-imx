load("//tensorflow/java:build_defs.bzl", "JAVACOPTS")
load("//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_jni_library")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")
load("@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_shared_library", "tflite_flex_cc_library", "generate_flex_kernel_header", "load_flex_kernel_header")

# Following targets are using for testing selective-built flex delegate
# in Java. Please don't use them for other purposes.

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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


# Following targets are used to generate a selective-built flex delegate 
# for elinux configurations, leveraging benchmark_model as an example.
#
# For reduced size targets, the build is done in 2 phases
#   1) Using native tools, generate the Flex listing and header
#       bazel [...] run //tensorflow/lite/delegates/flex/test:fetch_flex_files
#   2) Using the target tools, build the dynamic/static binaries
#       bazel [...] build -c opt --cxxopt=--std=c++17 --config=monolithic --config=elinux_aarch64 \
#           --host_crosstool_top=@bazel_tools//tools/cpp:toolchain [...]  \
#           //tensorflow/lite/delegates/flex/test:benchmark_model_plus_flex_reduced
#       bazel [...] build -c opt --cxxopt=--std=c++17 --config=monolithic --config=elinux_aarch64 \
#           --host_crosstool_top=@bazel_tools//tools/cpp:toolchain [...]  \
#           //tensorflow/lite/delegates/flex/test:benchmark_model_plus_flex_dynamic_reduced

# Sample model name used by the tests
_SAMPLE_MODEL = "simple_flex_model_int8.tflite"
exports_files([_SAMPLE_MODEL])

# Set of variables to configure target names and paths
# _PACK_NAME:           Current package name.
# _HEADERS_DIR:         Name to store fetched files.
# _FKH_TARGET_PREFIX:   Name used in 'generate_flex_kernel_header', \
#                       which wraps the 2 genrules that creates \
#                       the listing and header files.
# _FKH_TARGET_SUFFIX:   Name used internally by native tools \
#                       generating the flex listing and header files.
# _FLEX_OPS_LIST_GENRULE_NAME: 
#                       Name of the genrule that creates the flex ops listing.
# _FLEX_OPS_HEADER_GENRULE_NAME:
#                       Name of the genrule that creates the flex ops header.
_PACK_NAME = package_name()
_HEADERS_DIR = "flex_kernel_headers"
_LOCAL_HEADERS_DIR = _PACK_NAME + "/" + _HEADERS_DIR
_FKH_TARGET_PREFIX = "fkh_gen"
_FKH_TARGET_SUFFIX = "_tf_generated_kernel_header"
_FKH_TARGET = _PACK_NAME + "/" + _FKH_TARGET_PREFIX + _FKH_TARGET_SUFFIX
_FLEX_OPS_LIST_GENRULE_NAME = _FKH_TARGET_PREFIX + "_list_flex_ops"
_FLEX_OPS_HEADER_GENRULE_NAME = _FKH_TARGET_PREFIX + "_kernel_registration"

# Map between target files and genrules.
_GENERATED = {
    "list_flex_ops": ":" + _FLEX_OPS_LIST_GENRULE_NAME,
    "ops_to_register.h": ":" + _FLEX_OPS_HEADER_GENRULE_NAME,
}

# Rule to setup flex op file genrules using the
# example model 
generate_flex_kernel_header(
    name = _FKH_TARGET_PREFIX,
    models = [_SAMPLE_MODEL],
    testonly = 0,
)

# Skylib macro to generate the fetch script which copies
# the flex ops files. It uses the map between files and targets.
write_file(
    name = "fetch_flex_files_script",
    out = "fetch_flex_files.sh",
    content = [
        "#!/usr/bin/env bash",
        "cd $BUILD_WORKSPACE_DIRECTORY",
    ] + [
        "mkdir -p {0} && cp -fv bazel-bin/{2} {1}".format(
            _LOCAL_HEADERS_DIR,
            _LOCAL_HEADERS_DIR + "/" + k,
            _FKH_TARGET + "/" + k
        )
        for [k, _] in _GENERATED.items()
    ],
)

# Rule used to extract the flex op files into package path
sh_binary(
    name = "fetch_flex_files",
    srcs = ["fetch_flex_files.sh"],
    data = _GENERATED.values(),
)

# Rule outputing a struct containing the paths to the generated 
# flex ops files in the current package.
#   Note: Storing outside bazel context and loading the files 
#       acts as the interface to avoid building the tooling with 
#       the target platform toolchain.
loaded_headers = load_flex_kernel_header(
    include_path = _HEADERS_DIR,
    header_path = _HEADERS_DIR + "/ops_to_register.h",
)

# Rule to generate the reduced size - statically linked flex delegate
tflite_flex_cc_library(
    name = "delegate_reduced",
    models = [_SAMPLE_MODEL],
    visibility = ["//visibility:public"],
    kernel_headers = loaded_headers,
)

# Rule to generate the reduced size - statically linked `benchmark_model` bin
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

# Rule to generate the reduced size, dynamically linked flex delegate
tflite_flex_shared_library(
  name = "tensorflowlite_flex_reduced",
  models = [_SAMPLE_MODEL],
  visibility = ["//visibility:public"],
  kernel_headers = loaded_headers,
)

cc_import(
    name = "libtensorflowlite_flex_reduced",
    shared_library = ":tensorflowlite_flex_reduced",
)

# Rule to generate the reduced size - dynamically linked `benchmark_model` bin
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

# Rule to generate the full size - statically linked flex delegate
tflite_flex_cc_library(
    name = "delegate_full",
    visibility = ["//visibility:public"],
)

# Rule to generate the full size - statically linked `benchmark_model` bin
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

# Rule to generate the full size, dynamically linked flex delegate
tflite_flex_shared_library(
  name = "tensorflowlite_flex_full",
  visibility = ["//visibility:public"]
)

cc_import(
    name = "libtensorflowlite_flex_full",
    shared_library = ":tensorflowlite_flex_full",
)

# Rule to generate the full size - dynamically linked `benchmark_model` bin
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