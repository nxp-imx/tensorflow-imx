#
# Copyright 2020 NXP. All Rights Reserved.
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

# The -mcpu switch might be introduced externaly into CMake. This switch causes the
# XNNPACK compilation break. This switch needs to be removed for XNNPACK
# In order to isolate the changes only for XNNPACK and its depencencies, this subfolder is
# introduced.
string(REGEX REPLACE "-mcpu=[-a-zA-Z0-9_]*" "" _asm_flags ${CMAKE_ASM_FLAGS})
string(REGEX REPLACE "-mcpu=[-a-zA-Z0-9_]*" "" _c_flags ${CMAKE_C_FLAGS})
string(REGEX REPLACE "-mcpu=[-a-zA-Z0-9_]*" "" _cxx_flags ${CMAKE_CXX_FLAGS})

set(CMAKE_ASM_FLAGS ${_asm_flags})
set(CMAKE_C_FLAGS ${_c_flags})
set(CMAKE_CXX_FLAGS ${_cxx_flags})

add_subdirectory(
  "${xnnpack_SOURCE_DIR}"
  "${xnnpack_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)