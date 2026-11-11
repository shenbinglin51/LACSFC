#!/bin/bash
set -e

CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra"
INCLUDES="-I. -I../../include -I../../build/include"
LIBS="-L../../build/lib -L../../build/external/libdivsufsort/lib -lsdsl -ldivsufsort -ldivsufsort64"

echo "[ac_brq_nd] compiling ..."
$CXX $CXXFLAGS $INCLUDES \
    ac_brq_nd_example.cpp \
    -o ac_brq_nd_example \
    $LIBS
