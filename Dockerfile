FROM nvcr.io/nvidia/tensorflow:19.10-py3

# to let ubuntu install software without invoking questions
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
        && apt-get install -y --no-install-recommends \
            build-essential \
            ca-certificates \
            libssl-dev \
            curl \
            git \
            wget \
            python3-dev \
            python3-numpy \
            python3-setuptools \
            python3-scipy \
            python3-lmdb\
            libatlas-base-dev \
            libboost-all-dev \
            libcgal-dev \
            libeigen3-dev \
            libgflags-dev \
            libgoogle-glog-dev \
            libhdf5-serial-dev \
            libleveldb-dev \
            liblmdb-dev \
            libopencv-dev \
            libprotobuf-dev \
            libsnappy-dev \
            protobuf-compiler \
            rsync \
            software-properties-common \
            vim \
            tar \
            zip \
        && rm -rf /var/lib/apt/lists/*

RUN rm -r /usr/bin/python && ln -s /usr/bin/python3.6 /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py \
        && python get-pip.py \
        && rm get-pip.py

RUN pip3 --no-cache-dir install \
        scikit-build \
        Cython 

RUN pip3 install yacs tqdm

RUN  pip3 install -U gast==0.2.2 numpy==1.16.4

RUN mkdir ~/temp \
        && cd ~/temp \
        && wget https://github.com/Kitware/CMake/releases/download/v3.18.0/cmake-3.18.0.tar.gz \
        && tar -zxvf cmake-3.18.0.tar.gz \
        && cd cmake-3.18.0/ \
        && ./bootstrap \
        && make -j8 \
        && make install


WORKDIR /data
COPY m40_test_points.tfrecords .
COPY m40_train_points.tfrecords .

WORKDIR /code
RUN git clone https://github.com/microsoft/O-CNN.git

WORKDIR /code/O-CNN
RUN git checkout 316a85fe427fa21631ea61b8d72f894b051e57a7

COPY build_repo.sh .
RUN sh build_repo.sh


#WORKDIR /code/O-CNN
#RUN rm -rf *
#
#COPY caffe caffe
#COPY octree octree
#COPY tensorflow tensorflow
#COPY build_repo.sh .
#
#RUN sh build_repo.sh
#
#ENV SEG_DATA_DIR=/data/seg_data
#ENV SEG_OUT_DIR=/data/seg_out
#
#RUN cd tensorflow/script \
#    && echo python run_seg_partnet.py \
#        --config configs/segmentation/seg_hrnet_partnet_pts.yaml \
#        SOLVER.run train SOLVER.gpu 0, \
#        SOLVER.logdir ${SEG_OUT_DIR}/Chair/hrnet \
#        SOLVER.max_iter 20000 \
#        SOLVER.test_iter 1217 \
#        SOLVER.ckpt '' \
#        DATA.train.location ${SEG_DATA_DIR}/Chair_train_level3.tfrecords \
#        DATA.test.location ${SEG_DATA_DIR}/Chair_test_level3.tfrecords \
#        MODEL.nout 39 \
#        MODEL.factor 2 \
#        LOSS.num_class 39 \
#        DATA.train.take 4489 \
#        DATA.test.batch_size 1 \
#        DATA.train.batch_size 4

# commands to run:
# docker build -t dezavou/ocnntf .
# docker run -it dezavou/ocnntf /bin/bash
# docker run:
#     -d = --detach i.e. run in background
#     -p 6006:6006 = --publish 6006:6006 i.e. send traffic from 6006 to local 6006
#     -v /source/path:/dest/path = --mount source=/source/path,destination=/dest/path i.e. mount a volume 
# or git clone the original .. gives issue .. try with changing math import...

# docker build -t dezavou/ocnntfcommit .
# docker run -it -v /media/graphicslab/BigData/zavou/ANNFASS_data:/data dezavou/ocnntfcommit /bin/bash
