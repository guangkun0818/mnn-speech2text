#!/bin/bash

# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.08.30

mkdir -p build

cd build && cmake .. \
    -DMNN_BUILD_CONVERTER=ON # Build mnn converter.

make -j4