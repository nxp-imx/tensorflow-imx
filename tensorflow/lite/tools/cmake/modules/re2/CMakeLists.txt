# Copyright 2015 The RE2 Authors.  All Rights Reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Old enough to support Ubuntu Xenial.
cmake_minimum_required(VERSION 3.5.1)

if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

project(re2 CXX)
include(CTest)
include(GNUInstallDirs)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

option(BUILD_SHARED_LIBS "build shared libraries" OFF)
option(USEPCRE "use PCRE in tests and benchmarks" OFF)

# CMake seems to have no way to enable/disable testing per subproject,
# so we provide an option similar to BUILD_TESTING, but just for RE2.
option(RE2_BUILD_TESTING "enable testing for RE2" OFF)

set(RE2_SOURCE_DIR "" CACHE PATH
  "Directory that contains the Re2 project"
)
message(DEBUG "RE2 source directory: ${RE2_SOURCE_DIR}")
if(NOT RE2_SOURCE_DIR)
  message(FATAL_ERROR "Must specify source directory")
endif()

set(EXTRA_TARGET_LINK_LIBRARIES)

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  if(MSVC_VERSION LESS 1900)
    message(FATAL_ERROR "you need Visual Studio 2015 or later")
  endif()
  if(BUILD_SHARED_LIBS)
    # See http://www.kitware.com/blog/home/post/939 for details.
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
  # CMake defaults to /W3, but some users like /W4 (or /Wall) and /WX,
  # so we disable various warnings that aren't particularly helpful.
  add_compile_options(/wd4100 /wd4201 /wd4456 /wd4457 /wd4702 /wd4815)
  # Without a byte order mark (BOM), Visual Studio assumes that the source
  # file is encoded using the current user code page, so we specify UTF-8.
  add_compile_options(/utf-8)
endif()

if(WIN32)
  add_definitions(-DUNICODE -D_UNICODE -DSTRICT -DNOMINMAX)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
elseif(UNIX)
  add_compile_options(-pthread)
  list(APPEND EXTRA_TARGET_LINK_LIBRARIES -pthread)
endif()

if(USEPCRE)
  add_definitions(-DUSEPCRE)
  list(APPEND EXTRA_TARGET_LINK_LIBRARIES pcre)
endif()

include_directories(${RE2_SOURCE_DIR})
include_directories("${RE2_SOURCE_DIR}/../abseil-cpp/")

set(RE2_SOURCES
    ${RE2_SOURCE_DIR}/re2/bitstate.cc
    ${RE2_SOURCE_DIR}/re2/compile.cc
    ${RE2_SOURCE_DIR}/re2/dfa.cc
    ${RE2_SOURCE_DIR}/re2/filtered_re2.cc
    ${RE2_SOURCE_DIR}/re2/mimics_pcre.cc
    ${RE2_SOURCE_DIR}/re2/nfa.cc
    ${RE2_SOURCE_DIR}/re2/onepass.cc
    ${RE2_SOURCE_DIR}/re2/parse.cc
    ${RE2_SOURCE_DIR}/re2/perl_groups.cc
    ${RE2_SOURCE_DIR}/re2/prefilter.cc
    ${RE2_SOURCE_DIR}/re2/prefilter_tree.cc
    ${RE2_SOURCE_DIR}/re2/prog.cc
    ${RE2_SOURCE_DIR}/re2/re2.cc
    ${RE2_SOURCE_DIR}/re2/regexp.cc
    ${RE2_SOURCE_DIR}/re2/set.cc
    ${RE2_SOURCE_DIR}/re2/simplify.cc
    #${RE2_SOURCE_DIR}/re2/stringpiece.cc
    ${RE2_SOURCE_DIR}/re2/tostring.cc
    ${RE2_SOURCE_DIR}/re2/unicode_casefold.cc
    ${RE2_SOURCE_DIR}/re2/unicode_groups.cc
    ${RE2_SOURCE_DIR}/util/rune.cc
    ${RE2_SOURCE_DIR}/util/strutil.cc
    )

add_library(re2 ${RE2_SOURCES})
add_library(re2::re2 ALIAS re2)


