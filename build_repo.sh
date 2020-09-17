#!/bin/bash
#mkdir octree/external
# source activate OCNN

cd octree/external && git clone --recursive https://github.com/wang-ps/octree-ext.git
cd .. && mkdir build && cd build
cmake ..  && cmake --build . --config Release
export PATH=`pwd`:$PATH
cmake .. -DUSE_CUDA=ON && make
cd ../../tensorflow/libs && python build.py

echo "Below is the path"
echo $PATH

cd ../script && python run_cls.py \
        --config configs/cls_points.yaml \
        SOLVER.run train \
        SOLVER.gpu 0, \
        SOLVER.logdir /data/dockerCommitTrial \
        SOLVER.ckpt '' \
        DATA.train.location /data/m40_train_points.tfrecords \
        DATA.test.location /data/m40_test_points.tfrecords
