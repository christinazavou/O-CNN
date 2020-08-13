#!/bin/bash

repo=$(pwd)

cd /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/octree/build

./check_octree --filenames ${repo}/check_octrees_in.txt >${repo}/check_octrees_out.txt
