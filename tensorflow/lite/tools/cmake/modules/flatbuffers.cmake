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

if(TARGET flatbuffers OR flatbuffers_POPULATED)
  return()
endif()

# For flatbuffers to be built as part of host tools build (see tensorflow/lite/tools/cmake/native_tools)
if(NOT TF_SOURCE_DIR)
  get_filename_component(TF_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../tensorflow"
    ABSOLUTE
  )
endif()

include(utils)
get_dependency_archive("flatbuffers" "${TF_SOURCE_DIR}/../third_party/flatbuffers/workspace.bzl" FLATBUFFERS_URL FLATBUFFERS_CHECKSUM)
include(OverridableFetchContent)

OverridableFetchContent_Declare(
  flatbuffers
  URL ${FLATBUFFERS_URL}
  URL_HASH SHA256=${FLATBUFFERS_CHECKSUM}
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/flatbuffers"
)

OverridableFetchContent_GetProperties(flatbuffers)
if(NOT flatbuffers_POPULATED)
  OverridableFetchContent_Populate(flatbuffers)
endif()

option(FLATBUFFERS_BUILD_TESTS OFF)
# Required for Windows, since it has macros called min & max which
# clashes with std::min
add_definitions(-DNOMINMAX=1)
message(STATUS "flatbuffers_source_dir ${flatbuffers_SOURCE_DIR}")
message(STATUS "flatbuffers_binary_dir ${flatbuffers_BINARY_DIR}")

add_subdirectory(
  "${flatbuffers_SOURCE_DIR}"
  "${flatbuffers_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)
remove_definitions(-DNOMINMAX)

# For BuildFlatBuffers.cmake
set(CMAKE_MODULE_PATH
  "${flatbuffers_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
)

# The host-side flatc binary
include(ExternalProject)

# For native flatc build purposes the flatc needs to be included in 'all' target
if(NOT DEFINED FLATC_EXCLUDE_FROM_ALL)
  set(FLATC_EXCLUDE_FROM_ALL TRUE)
endif()

# In case of a standalone (native) build of flatc for unit test cross-compilation
# purposes the FLATC_INSTALL_PREFIX is already in cache and is just used in this module.
# In case of standard flatbuffers build (as a dependency) the variable needs to be set. 
if(NOT DEFINED FLATC_INSTALL_PREFIX)
  set(FLATC_INSTALL_PREFIX <INSTALL_DIR> CACHE PATH "Flatc installation directory")
endif()

ExternalProject_Add(flatbuffers-flatc
  PREFIX ${CMAKE_BINARY_DIR}/flatbuffers-flatc
  SOURCE_DIR ${flatbuffers_SOURCE_DIR}
  CMAKE_ARGS -DCMAKE_CXX_FLAGS="-DNOMINMAX=1"
             -DFLATBUFFERS_BUILD_TESTS=OFF
             -DFLATBUFFERS_BUILD_FLATLIB=OFF
             -DFLATBUFFERS_STATIC_FLATC=OFF
             -DFLATBUFFERS_BUILD_FLATHASH=OFF
             -DCMAKE_INSTALL_PREFIX=$CACHE{FLATC_INSTALL_PREFIX}
             -DCMAKE_DISABLE_FIND_PACKAGE_Python3=TRUE
  EXCLUDE_FROM_ALL ${FLATC_EXCLUDE_FROM_ALL}
)
