#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

if(TARGET xnnpack OR xnnpack_POPULATED)
  return()
endif()

include(utils)
get_dependency_tag("xnnpack" "${TF_SOURCE_DIR}/workspace2.bzl" XNNPACK_TAG)

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  xnnpack
  GIT_REPOSITORY https://github.com/google/XNNPACK
  GIT_TAG ${XNNPACK_TAG}
  GIT_PROGRESS TRUE
  PREFIX "${CMAKE_BINARY_DIR}"
  SOURCE_DIR "${CMAKE_BINARY_DIR}/xnnpack"
  PATCH_COMMAND patch -p1 --forward < ${CMAKE_CURRENT_LIST_DIR}/xnnpack/patches/0001-Partial-changes-for-fixing-runtime-setup-issue.patch || true
)
OverridableFetchContent_GetProperties(xnnpack)
if(NOT xnnpack_POPULATED)
  OverridableFetchContent_Populate(xnnpack)
endif()

# May consider setting XNNPACK_USE_SYSTEM_LIBS if we want to control all
# dependencies by TFLite.
set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "Disable XNNPACK test.")
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "Disable XNNPACK benchmarks.")

# The following line adds project of PTHREADPOOL, FP16 and XNNPACK which are
# needed to compile XNNPACK delegate of TFLite.
add_subdirectory(
  "${xnnpack_SOURCE_DIR}"
  "${xnnpack_BINARY_DIR}"
)

include_directories(
  AFTER
   "${PTHREADPOOL_SOURCE_DIR}/include"
   "${FP16_SOURCE_DIR}/include"
   "${XNNPACK_SOURCE_DIR}/include"
   "${CPUINFO_SOURCE_DIR}/"
)
