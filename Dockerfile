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


WORKDIR /code/O-CNN
RUN rm -rf *

COPY caffe caffe
COPY octree octree
COPY tensorflow tensorflow
COPY build_repo.sh .

RUN sh build_repo.sh
