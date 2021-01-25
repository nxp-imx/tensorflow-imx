
if(TARGET re2 OR re2_POPULATED)
    return()
endif()

include(OverridableFetchContent)

OverridableFetchContent_Declare(
  re2
  URL https://github.com/google/re2/archive/506cfa4bffd060c06ec338ce50ea3468daa6c814.tar.gz
  URL_HASH SHA256=d070e2ffc5476c496a6a872a6f246bfddce8e7797d6ba605a7c8d72866743bf9
  
  SOURCE_DIR "${CMAKE_BINARY_DIR}/re2"
)
    
OverridableFetchContent_GetProperties(re2)
if(NOT re2_POPULATED)
  OverridableFetchContent_Populate(re2)
endif()

set(RE2_SOURCE_DIR ${re2_SOURCE_DIR} CACHE PATH "Directory that contains the Re2 project" FORCE)
message(DEBUG "RE2 source directory: ${RE2_SOURCE_DIR}")

add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/re2"
  "${re2_BINARY_DIR}"
  EXCLUDE_FROM_ALL
)

include_directories(
  AFTER
  ${RE2_SOURCE_DIR}
)