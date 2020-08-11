#!/bin/bash

cd /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/octree/build

mkdir /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/mycode/test/output/ocnn_completion/points2ply

./points2ply --filenames /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/mycode/test/resources/ocnn_completion_only2samples2/points2ply.txt --output_path /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/mycode/test/output/ocnn_completion/points2ply
