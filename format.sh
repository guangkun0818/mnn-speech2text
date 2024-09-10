#!/bin/bash

# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.30

# Code format script

# C++ runtime clang-format 16.0.0
find ./mnn-s2trt/ -iname "*.h" -o \
    -iname "*.cc" -o -iname "*.c" \
    -o -iname "*.cpp" | xargs clang-format -style=Google -i

# CMakes cmake-format version 0.6.13
find ./cmake/ -iname "*.cmake" | xargs cmake-format -i
find ./ -path "./3rd_party" -prune -o \
    -path "./build" -prune -o \
    -iname "CMakeLists.txt" -print | xargs cmake-format -i