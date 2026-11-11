#!/bin/bash
set -e

CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra"

echo "[ac_brq_nd_space_report] compiling ..."
$CXX $CXXFLAGS \
    ac_brq_nd_space_report.cpp \
    -o ac_brq_nd_space_report
