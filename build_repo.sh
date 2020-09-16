#!/bin/bash
mkdir octree/external
cd octree/external && git clone --recursive https://github.com/wang-ps/octree-ext.git
cd .. && mkdir build && cd build
# source activate OCNN
cmake ..  && cmake --build . --config Release
export PATH=`pwd`:$PATH
cmake .. -DUSE_CUDA=ON && make
cd ../../tensorflow/libs && python build.py
