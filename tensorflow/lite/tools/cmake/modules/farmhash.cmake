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

if(TARGET farmhash OR farmhash_POPULATED)
  return()
endif()

include(utils)
get_dependency_tag("farmhash" "${TF_SOURCE_DIR}/../third_party/farmhash/workspace.bzl" FARMHASH_TAG)

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  farmhash
  GIT_REPOSITORY https://github.com/google/farmhash
  GIT_TAG ${FARMHASH_TAG}
  # It's not currently possible to shallow clone with a GIT TAG
  # as cmake attempts to git checkout the commit hash after the clone
  # which doesn't work as it's a shallow clone hence a different commit hash.
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17770
  # GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/farmhash"
)
OverridableFetchContent_GetProperties(farmhash)
if(NOT farmhash_POPULATED)
  OverridableFetchContent_Populate(farmhash)
endif()

set(FARMHASH_SOURCE_DIR "${farmhash_SOURCE_DIR}" CACHE PATH
  "Source directory for the CMake project."
)

add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/farmhash"
  "${farmhash_BINARY_DIR}"
)
