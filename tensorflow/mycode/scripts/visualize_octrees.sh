#!/bin/bash

out_dir='/media/christina/Data/ANFASS_data/O-CNN/output/ModelNet40/logsT3/misclassified'
prob='0.9'
mkdir ${out_dir}
mkdir ${out_dir}/probability_${prob}

cd ../../../octree/build

./octree2mesh --filenames ${cur_dir}/visualize_octrees_paths.txt --output_path ${out_dir}/probability_${prob}
