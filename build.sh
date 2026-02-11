#!/bin/bash

# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.30

mkdir -p build

# Build with mnn converter.
cd build && cmake .. \
  -DMNN_BUILD_CONVERTER=ON \
  -DCMAKE_COMPILE_COMMANDS=ON

make -j32