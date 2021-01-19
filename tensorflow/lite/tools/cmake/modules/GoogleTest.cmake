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

if(TARGET GoogleTest OR GoogleTest_POPULATED)
    return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  GoogleTest
  URL https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip
  URL_HASH SHA256=ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86
  
  SOURCE_DIR "${CMAKE_BINARY_DIR}/googletest"
)
    
OverridableFetchContent_GetProperties(GoogleTest)
if(NOT GoogleTest_POPULATED)
  OverridableFetchContent_Populate(GoogleTest)
endif()

add_subdirectory(
  "${googletest_SOURCE_DIR}"
  "${googletest_BINARY_DIR}"
)

include_directories(
  AFTER
  "${googletest_SOURCE_DIR}/googletest/include"
  "${googletest_SOURCE_DIR}/googlemock/include"
)