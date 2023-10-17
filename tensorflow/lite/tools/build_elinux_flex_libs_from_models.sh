#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2023 NXP. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to build Flex delegate libraries (static or dynamic) from given models
# Optionally, a test application (benchmark_model) will be built for testing purposes.
# Note: Use from the docker image and tag described in the ML User Guide
# ==============================================================================

set -e


BZL_OUTPUT_INTER_PATH="execroot/org_tensorflow/bazel-out/aarch64-opt/bin"


function print_usage {
  echo "Usage:"
  echo "  $(basename ${BASH_SOURCE}) \\"
  echo "    [--input_models=model1.tflite,model2.tflite...] \\"
  echo "    [--static_link] \\"
  echo "    [--build_libs | --build_benchmark_app] \\"
  echo ""
  echo "Where: "
  echo "  --input_models: TFLite models to extract Flex ops from. "
  echo "  --static_link: Set static linkage (default dynamic)."
  echo ""
  echo "Build arguments (mutually exclusive). Latest argument takes priority:"
  echo "Without at least one of these arguments, only Bazel BUILD files are generated."
  echo "  --build_libs: Build flex libs for given models."
  echo "  --build_benchmark_app: Build benchmark_model app."
  echo "                         Use it as an example/minimal template."
  echo ""
  exit 1
}


function generate_list_field {
  local name="$1"
  local list_string="$2"
  local list=(${list_string//,/ })

  local message=("$name=[")
  for item in "${list[@]}" ; do
    message+=("\"$item\",")
  done
  message+=('],')
  printf '%s' "${message[@]}"
}


function generate_flex_buildfiles {
  pushd ${TMP_DIR} > /dev/null
  # Generating the BUILD file.

  if [ ${STATIC_LINK_FLAG} = true ]; then
    message=('# Generated Bazel file to build static Flex library')
  else
    message=('# Generated Bazel file to build dynamic Flex library')
  fi
  message+=(
    '# Source: '$(basename "$0")
    ''
    'load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")'
    'load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")'
  )
  if [ ${STATIC_LINK_FLAG} = true ]; then
    message+=(
      'load("@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_cc_library")'
      ''
      '# Rule to generate static Flex lib'
      'tflite_flex_cc_library('
      '    name = "tensorflowlite_flex",'
      )
      if [ ! -z ${MODEL_NAMES} ]; then
        message+=('    '$(generate_list_field "models" $MODEL_NAMES))
      fi
      message+=(
        ')'
      )
  else
    message+=(
      'load("@org_tensorflow//tensorflow/lite/delegates/flex:build_def.bzl", "tflite_flex_shared_library")'
      ''
      '# Rule to generate dynamic Flex lib'
      'tflite_flex_shared_library('
      '    name = "tensorflowlite_flex_dynamic",'
      )
    if [ ! -z ${MODEL_NAMES} ]; then
      message+=('    '$(generate_list_field "models" $MODEL_NAMES))
    fi
    message+=(
      ')'
      ''
      'cc_import('
      '    name = "libtensorflowlite_flex_dynamic",'
      '    shared_library = ":tensorflowlite_flex_dynamic",'
      ')'
    )
  fi

  if [ ${BUILD_BENCHMARK} = true ]; then
      if [ ${STATIC_LINK_FLAG} = true ]; then
        message+=(
          ''
          '# Rule to build benchmark_model application with static flex lib'
          '# Feel free to edit this rule to include the developed elements (sources, other deps, etc)'
          'tf_cc_binary('
          '    name = "benchmark_model_plus_flex",'
          '    srcs = ['
          '        "//tensorflow/lite/tools/benchmark:benchmark_plus_flex_main.cc",'
          '    ],'
          '    copts = tflite_copts() + tflite_copts_warnings(),'
          '    linkopts = tflite_linkopts(),'
          '    deps = ['
          '        ":tensorflowlite_flex",'
          '        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",'
          '        "//tensorflow/lite/testing:init_tensorflow",'
          '        "//tensorflow/lite/tools:logging",'
          '    ],'
          ')'
        )
      else
        message+=(
          ''
          '# Rule to build benchmark_model application with dynamic flex lib'
          '# Feel free to edit this rule to include the developed elements (sources, other deps, etc)'
          'tf_cc_binary('
          '    name = "benchmark_model_plus_flex_dynamic",'
          '    srcs = ['
          '        "//tensorflow/lite/tools/benchmark:benchmark_plus_flex_main.cc",'
          '    ],'
          '    copts = tflite_copts() + tflite_copts_warnings(),'
          '    linkopts = tflite_linkopts(),'
          '    deps = ['
          '        ":libtensorflowlite_flex_dynamic",'
          '        "//tensorflow/lite/tools/benchmark:benchmark_tflite_model_lib",'
          '        "//tensorflow/lite/testing:init_tensorflow",'
          '        "//tensorflow/lite/tools:logging",'
          '    ],'
          ')'
        )
      fi
  fi
  printf '%s\n' "${message[@]}" >> BUILD
  popd > /dev/null

  echo "[INFO] Generated bazel file in ${TMP_DIR}"
  echo "**************************************************************************************************"
  cat ${TMP_DIR}/BUILD
  echo "**************************************************************************************************"

}


function build_libs {
  if [ ${STATIC_LINK_FLAG} = true ]; then
    bazel --output_base=${BZL_OUTPUT_BASE} build \
      -c opt --cxxopt='--std=c++17' \
      --config=monolithic \
      --config=elinux_aarch64 \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex

    echo "[INFO] Rule 'tensorflowlite_flex' generation ended. Note that objects are located in bazel-bin dir, " \
      "but not archived. Include 'tensorflowlite_flex' as 'deps' in target application rule. For more info, " \
      "run this script with '--build_benchmark_app' flag and check/edit the dependencies."
  else
    bazel --output_base=${BZL_OUTPUT_BASE} build \
      -c opt --cxxopt='--std=c++17' \
      --config=monolithic \
      --config=elinux_aarch64 \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:tensorflowlite_flex_dynamic

    OUT_FILES="${OUT_FILES} ${BZL_OUTPUT_BASE}/${BZL_OUTPUT_INTER_PATH}/tmp/libtensorflowlite_flex_dynamic.so"
  fi
}


function build_benchmark_app {
  if [ ${STATIC_LINK_FLAG} = true ]; then
    bazel --output_base=${BZL_OUTPUT_BASE} build \
        -c opt --cxxopt='--std=c++17' \
        --config=monolithic \
        --config=elinux_aarch64 \
        --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
        //tmp:benchmark_model_plus_flex

    OUT_FILES="${OUT_FILES} ${BZL_OUTPUT_BASE}/${BZL_OUTPUT_INTER_PATH}/tmp/benchmark_model_plus_flex"
  else
    bazel --output_base=${BZL_OUTPUT_BASE} build \
      -c opt --cxxopt='--std=c++17' \
      --config=monolithic \
      --config=elinux_aarch64 \
      --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
      //tmp:benchmark_model_plus_flex_dynamic

    OUT_FILES="${OUT_FILES} ${BZL_OUTPUT_BASE}/${BZL_OUTPUT_INTER_PATH}/tmp/benchmark_model_plus_flex_dynamic"
  fi
}

#######################################################################################################################

# Check if launcher or manual execution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d /script_dir ]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../tensorflow" && pwd)"
  BZL_OUTPUT_BASE="/tensorflow/docker-build"
else
  echo "[WARNING] Running outside docker. In cases of errors, please try the docker image."
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  BZL_OUTPUT_BASE="${ROOT_DIR}/bazel-base-build"
fi

# Check & parse inputs
STATIC_LINK_FLAG=false
BUILD_BENCHMARK=false
BUILD_LIBS=false

if [ "$#" -gt 4 ]; then
  echo "[ERROR] Too many arguments."
  print_usage
fi

for i in "$@"; do
  case $i in
      --input_models=*) 
        FLAG_MODELS="${i#*=}" 
        shift;;
      --static_link) 
        STATIC_LINK_FLAG=true 
        shift;;
      --build_libs)
        BUILD_LIBS=true 
        BUILD_BENCHMARK=false
        shift;;
      --build_benchmark_app)
        BUILD_BENCHMARK=true 
        BUILD_LIBS=false
        shift;;
      *)
        echo "[ERROR] Unrecognized argument: ${i}"
        print_usage;;
  esac
done

# Run & Check if users already run configure
cd $ROOT_DIR
if [ ! -f "$ROOT_DIR/.tf_configure.bazelrc" ]; then
  echo "[WARNING] Running default configure. If changes needed, please manually run ./configure first."
  configs=(
    '/usr/bin/python3'
    '/usr/lib/python3/dist-packages'
    'N'
    'N'
    'N'
    'N'
    'N'
  )
  printf '%s\n' "${configs[@]}" | ./configure
fi

# Prepare the tmp directory.
TMP_DIR="${ROOT_DIR}/tmp/"
rm -rf ${TMP_DIR} && mkdir -p ${TMP_DIR}

# Copy models to tmp directory.
MODEL_NAMES=""
if [ ! -z ${FLAG_MODELS} ]; then
  for model in $(echo ${FLAG_MODELS} | sed "s/,/ /g"); do
    cp ${model} ${TMP_DIR}
    MODEL_NAMES="${MODEL_NAMES},$(basename ${model})"
  done
  if [ -z ${MODEL_NAMES} ]; then
    echo "[WARNING] Beware! No model was parsed, building the full Flex delegate." \
          "Note: check binary sizes, total is about ~100MB stripped"
  else
    echo "[INFO] Parsed models from arguments: ${MODEL_NAMES}" 
  fi
fi

# Generate & build
OUT_FILES=""

generate_flex_buildfiles

if [ ${BUILD_LIBS} = true ]; then
  build_libs
elif [ ${BUILD_BENCHMARK} = true ]; then
  build_benchmark_app
fi

# List the output files.
if [ ! -z "${OUT_FILES}" ]; then
  echo "[INFO] Output can be found here:"
  for out in ${OUT_FILES}; do
    ls -1a "${out}"
  done
fi
