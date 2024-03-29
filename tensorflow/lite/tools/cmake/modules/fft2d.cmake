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

if(TARGET fft2d OR fft2d_POPULATED)
  return()
endif()

include(utils)
get_dependency_archive("fft2d" "${TF_SOURCE_DIR}/workspace2.bzl" FFT2D_URL FFT2D_CHECKSUM)

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  fft2d
  URL ${FFT2D_URL}
  URL_HASH SHA256=${FFT2D_CHECKSUM}
  SOURCE_DIR "${CMAKE_BINARY_DIR}/fft2d"
  LICENSE_FILE "readme2d.txt"
  LICENSE_URL "http://www.kurims.kyoto-u.ac.jp/~ooura/fft.html"
)
OverridableFetchContent_GetProperties(fft2d)
if(NOT fft2d_POPULATED)
  OverridableFetchContent_Populate(fft2d)
endif()

set(FFT2D_SOURCE_DIR "${fft2d_SOURCE_DIR}" CACHE PATH "fft2d source")
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/fft2d"
  "${fft2d_BINARY_DIR}"
)
