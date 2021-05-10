#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

message("VX Delegate Enabled")

set(TFLITE_DELEGATES_VX_PUBLIC_OPTIONS "-DTFLITE_WITH_VX")

list(APPEND TFLITE_DELEGATES_VX_SRCS
  ${TFLITE_SOURCE_DIR}/delegates/vx-delegate/delegate_main.cc
  ${TFLITE_SOURCE_DIR}/delegates/vx-delegate/op_map.cc
  ${TFLITE_SOURCE_DIR}/delegates/vx-delegate/utils.cc
  ${TF_SOURCE_DIR}/core/platform/default/logging.cc
  ${TF_SOURCE_DIR}/core/platform/default/env_time.cc
)

if(NOT TIM_VX_INSTALL OR NOT EXISTS ${TIM_VX_INSTALL})
  message(FATAL_ERROR "Set TIM_VX_INSTALL to tim-vx install folder(by make install) instead of ${TIM_VX_INSTALL}")
endif()

message("=== Building with TIM_VX_LIBRIRIES from ${TIM_VX_INSTALL}")
include_directories(${TIM_VX_INSTALL}/include)
list(APPEND TFLITE_DELEGATES_VX_DEPENDENCIES ${TIM_VX_INSTALL}/lib/libtim-vx.so)