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

if(TARGET ruy OR ruy_POPULATED)
  return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  ruy
  URL https://github.com/google/ruy/archive/5bb02fbf90824c2eb6cd7418f766c593106a332b.zip
  URL_HASH SHA256=d8f9dc52c0a52c8470e2e0b60bc16cba91853d812846c075f7ed8404990b003d
  SOURCE_DIR "${CMAKE_BINARY_DIR}/ruy"
)
OverridableFetchContent_GetProperties(ruy)
if(NOT ruy_POPULATED)
  OverridableFetchContent_Populate(ruy)
endif()

set(RUY_SOURCE_DIR "${ruy_SOURCE_DIR}" CACHE PATH "RUY source directory")

add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/ruy"
  "${ruy_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)
