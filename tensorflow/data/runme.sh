#!/bin/bash
#SHELL := /bin/bash # otherwise we can't use "source"

prefix='/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour'

export PATH=/home/christina/miniconda3/bin/:$$PATH
source activate OCNN

# RUN COMMAND 1
#split_data_label_indices_in_files(prefix+'/dataset_points_chunk8/train.txt', True)

# RUN COMMAND 2
python seg_buildings_with_colour.py """write_data_to_tfrecords('${prefix}/w_colour_norm_w_labels', '${prefix}/dataset_points_chunk8/train.shuffle.0.txt', '${prefix}/dataset_points_chunk8/train.shuffle.0.tfrecords', 'data')""" > kati.txt
python seg_buildings_with_colour.py """write_data_to_tfrecords('${prefix}/w_colour_norm_w_labels', '${prefix}/dataset_points_chunk8/train.shuffle.1.txt', '${prefix}/dataset_points_chunk8/train.shuffle.1.tfrecords', 'data')""" >> kati.txt
python seg_buildings_with_colour.py """write_data_to_tfrecords('${prefix}/w_colour_norm_w_labels', '${prefix}/dataset_points_chunk8/train.shuffle.2.txt', '${prefix}/dataset_points_chunk8/train.shuffle.2.tfrecords', 'data')""" >> kati.txt
python seg_buildings_with_colour.py """write_data_to_tfrecords('${prefix}/w_colour_norm_w_labels', '${prefix}/dataset_points_chunk8/train.shuffle.3.txt', '${prefix}/dataset_points_chunk8/train.shuffle.3.tfrecords', 'data')""" >> kati.txt

# RUN COMMAND 3
#cat train.shuffle.0.tfrecords > train.shuffle.all.tfrecords
#cat train.shuffle.1.tfrecords >> train.shuffle.all.tfrecords
#cat train.shuffle.2.tfrecords >> train.shuffle.all.tfrecords
#cat train.shuffle.3.tfrecords >> train.shuffle.all.tfrecords
