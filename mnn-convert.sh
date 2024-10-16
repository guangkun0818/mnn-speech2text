#!/bin/bash

# Author: guangkun0818
# Email: 609946862@qq.com
# Created on 2024.09.30

onnx_model=$1
mnn_model=$2

./3rd_party/mnn-build/MNNConvert \
    --framework=ONNX \
    --modelFile=${onnx_model} \
    --MNNModel=${mnn_model} \
    --weightQuantBits=8