#!/bin/bash

RED='\033[0;31m'
NC='\033[0m' # No Color


build_with_cmake() {
  local usage="Usage: build_with_cmake folder"
  local lib="${1:?${usage}}"
  echo -e "${RED}Building $lib library${NC}"
  mkdir $lib/build 
  cd $lib/build
  cmake -DCMAKE_TOOLCHAIN_FILE=${OE_CMAKE_TOOLCHAIN_FILE} ..
  make -j4
  echo -e "${RED}Building $lib finished${NC}"
  cd ../..
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo $SCRIPT_DIR


cd $SCRIPT_DIR/downloads
cp ../re2.CMakeLists.txt re2/CMakeLists.txt
build_with_cmake "re2"
build_with_cmake "googletest"
build_with_cmake "absl"

