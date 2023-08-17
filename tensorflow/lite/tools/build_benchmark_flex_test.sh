#!/bin/bash
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
# Script to build Flex delegate test application (benchmark)
# Note: Use from the docker image and tag described in the User Guide
# ==============================================================================

set -e

# Check if launcher or manual execution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d /script_dir ]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../tensorflow" && pwd)"
  BZL_OUTPUT_BASE="/tensorflow/docker-build"
else
  echo "[Warning] Running outside docker. In cases of errors, please try the docker image."
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  BZL_OUTPUT_BASE="${ROOT_DIR}/bazel-base-build"
fi

function print_usage {
  echo "Usage:"
  echo "  $(basename ${BASH_SOURCE}) \\"
  echo "    [--reduced_size] \\"
  echo "    [--dynamic_link] \\"
  echo ""
  echo "Where: "
  echo "  --reduced_size: set build non-TFlite ops present in model (default all non-TFLite ops aka full size)."
  echo "  --dynamic_link: set dynamic linkage (default static)."
  echo ""
  echo "Note:"
  echo "    In case of finding any issues, remember that execution from docker is preferred."
  echo "    Further details are described in the latest i.MX Machine Learning User's Guide."
  exit 1
}

FLAG_REDUCED=false
FLAG_DYN_LINK=false

# Target string build
BZL_OUTPUT_INTER_PATH="execroot/org_tensorflow/bazel-out/aarch64-opt/bin/"
BZL_TARGET_LOC="tensorflow/lite/delegates/flex/test"
BZL_TARGET_BNAME="benchmark_model_plus_flex"


if [ "$#" -gt 2 ]; then
  echo "ERROR: Too many arguments."
  print_usage
fi

for i in "$@"
do
case $i in
    --reduced_size)
      FLAG_REDUCED=true
      shift;;
    --dynamic_link)
      FLAG_DYN_LINK=true
      shift;;
    *)
      echo "ERROR: Unrecognized argument: ${i}"
      print_usage;;
esac
done

# Run & Check if users already run configure
cd $ROOT_DIR
if [ ! -f "$ROOT_DIR/.tf_configure.bazelrc" ]; then
  echo "Warning: Running default configure. If changes needed, please manually run ./configure first."
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


# Build bazel target command
if [ ${FLAG_REDUCED} = true ]; then
  BZL_TARGET_SIZE="reduced"
else
  BZL_TARGET_SIZE="full"
fi

if [ ${FLAG_DYN_LINK} = true ]; then
  BZL_TARGET_LINK="dynamic_"
fi

BZL_TARGET="//${BZL_TARGET_LOC}:${BZL_TARGET_BNAME}_${BZL_TARGET_LINK}${BZL_TARGET_SIZE}"

# Trace output base and bazel target
echo "[Flex Delegate Test] Bazel target: ${BZL_TARGET}"
echo "[Flex Delegate Test] Output base: ${BZL_OUTPUT_BASE}"

# Build the target
bazel --output_base=${BZL_OUTPUT_BASE} build \
  -c opt --cxxopt='--std=c++17' \
  --config=monolithic \
  --config=elinux_aarch64 \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  ${BZL_TARGET}

echo "[Flex Delegate Test] bin out: "
ls -lh ${BZL_OUTPUT_BASE}/${BZL_OUTPUT_INTER_PATH}/${BZL_TARGET_LOC}/${BZL_TARGET_BNAME}_${BZL_TARGET_LINK}${BZL_TARGET_SIZE}
if [ ${FLAG_DYN_LINK} = true ]; then
  echo "[Flex Delegate Test] .so out: "
  ls -lh ${BZL_OUTPUT_BASE}/${BZL_OUTPUT_INTER_PATH}/${BZL_TARGET_LOC}/libtensorflowlite_flex_${BZL_TARGET_SIZE}.so
fi


