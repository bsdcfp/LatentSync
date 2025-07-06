#!/bin/bash

## 1. 编译Plugin
cd productscene/trt_backend/plugin

rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make

# Define the target directory
TARGET_DIR="../../../../models/tensorrt/T4"

# Check if the directory exists, and if not, create it
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
fi

# Copy all .so files to the target directory
cp *.so "$TARGET_DIR/"

cd ../../../..

## 2. 转换模型

python trt_backend/convert_to_trt.py --unet
python trt_backend/convert_to_trt.py --brushnet
python trt_backend/convert_to_trt.py --textencoder
python trt_backend/convert_to_trt.py --vae