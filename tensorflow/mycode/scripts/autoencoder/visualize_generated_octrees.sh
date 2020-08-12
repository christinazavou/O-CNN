#!/bin/bash

cd /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/octree/build

#logs_path="/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/logs/ae/ocnn_b16_decode_shape"
logs_path="/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/logs/ae/resnet_b16_decode_shape"

./octree2mesh --filenames ${logs_path}/input_shapes.txt --output_path ${logs_path}/inputshapes

./octree2mesh --filenames ${logs_path}/output_shapes.txt --output_path ${logs_path}/outputshapes
