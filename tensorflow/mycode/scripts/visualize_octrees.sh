#!/bin/bash
cur_dir=$(pwd)
mkdir misclassified

cd ../../../octree/build

./octree2mesh --filenames ${cur_dir}/visualize_octrees_paths.txt --output_path ${cur_dir}/misclassified
