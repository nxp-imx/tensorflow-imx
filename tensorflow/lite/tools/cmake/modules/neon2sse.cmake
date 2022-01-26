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

include(ExternalProject)

if(TARGET neon2sse OR neon2sse_POPULATED)
  return()
endif()

include(utils)
get_dependency_archive("neon2sse" "${TF_SOURCE_DIR}/workspace2.bzl" NEON2SSE_URL NEON2SSE_CHECKSUM)

OverridableFetchContent_Declare(
  neon2sse
  URL ${NEON2SSE_URL}
  URL_HASH SHA256=${NEON2SSE_CHECKSUM}
  SOURCE_DIR "${CMAKE_BINARY_DIR}/neon2sse"
)

OverridableFetchContent_GetProperties(neon2sse)
if(NOT neon2sse_POPULATED)
  OverridableFetchContent_Populate(neon2sse)
endif()

add_subdirectory(
  "${neon2sse_SOURCE_DIR}"
  "${neon2sse_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)
