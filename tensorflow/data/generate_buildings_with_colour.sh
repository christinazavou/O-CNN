#!/bin/bash
#SHELL := /bin/bash # otherwise we can't use "source"

prefix='/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour'
logfile='train_logs.txt'

export PATH=/home/christina/miniconda3/bin/:$$PATH
source activate OCNN

# split data into chunks of 500 otherwise memory explodes ...
python seg_buildings_with_colour.py """split_data_label_indices_in_files('${prefix}/dataset_points_chunk8/train.txt')""" > ${logfile}

# create tfrecords files for each chunk
python seg_buildings_with_colour.py """write_data_to_tfrecords('${prefix}/w_colour_norm_w_labels', '${prefix}/dataset_points_chunk8/train.shuffle.0.txt', '${prefix}/dataset_points_chunk8/train.shuffle.0.tfrecords', 'data')""" >> ${logfile}
python seg_buildings_with_colour.py """write_data_to_tfrecords('${prefix}/w_colour_norm_w_labels', '${prefix}/dataset_points_chunk8/train.shuffle.1.txt', '${prefix}/dataset_points_chunk8/train.shuffle.1.tfrecords', 'data')""" >> ${logfile}
python seg_buildings_with_colour.py """write_data_to_tfrecords('${prefix}/w_colour_norm_w_labels', '${prefix}/dataset_points_chunk8/train.shuffle.2.txt', '${prefix}/dataset_points_chunk8/train.shuffle.2.tfrecords', 'data')""" >> ${logfile}
python seg_buildings_with_colour.py """write_data_to_tfrecords('${prefix}/w_colour_norm_w_labels', '${prefix}/dataset_points_chunk8/train.shuffle.3.txt', '${prefix}/dataset_points_chunk8/train.shuffle.3.tfrecords', 'data')""" >> ${logfile}

# merge the tfrecords files
prefix=${prefix}/dataset_points_chunk8
cat ${prefix}/train.shuffle.0.tfrecords > ${prefix}/train.shuffle.all.tfrecords
cat ${prefix}/train.shuffle.1.tfrecords >> ${prefix}/train.shuffle.all.tfrecords
cat ${prefix}/train.shuffle.2.tfrecords >> ${prefix}/train.shuffle.all.tfrecords
cat ${prefix}/train.shuffle.3.tfrecords >> ${prefix}/train.shuffle.all.tfrecords
