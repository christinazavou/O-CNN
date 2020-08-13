#!/bin/bash

repo="/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/mycode"
resources="${repo}/test/resources/ocnn_completion_only2samples2"
outputs="${repo}/test/intermediate/ocnn_completion_only2samples2"
source="${repo}/src"

mkdir "${outputs}"
cp -r ${resources}/* ${outputs}

cd ${source}

python data_parsing.py """FileManipulator.generate_list_text_files('${outputs}/shape.points')"""

python data_parsing.py """FileManipulator.generate_octrees_for_each_folder('${outputs}/shape.points', '${outputs}/shape.octrees', '--depth 6 --split_label 1 --rot_num 6')"""

python data_parsing.py """TFRecordsConverter.write_records('${outputs}/shape.points', '${outputs}/filelist_test_points.txt', '${outputs}/test_points.tfrecords', file_type='data', shuffle=False)"""
python data_parsing.py """TFRecordsConverter.write_records('${outputs}/shape.points', '${outputs}/filelist_train_points.txt', '${outputs}/train_points.tfrecords', file_type='data', shuffle=False)"""

python data_parsing.py """FileManipulator.point_list_to_octree_list('${outputs}/filelist_test_points.txt', '${outputs}/filelist_test_octrees.txt', 6, 6)"""
python data_parsing.py """FileManipulator.point_list_to_octree_list('${outputs}/filelist_train_points.txt', '${outputs}/filelist_train_octrees.txt', 6, 6)"""

python data_parsing.py """TFRecordsConverter.write_records('${outputs}/shape.octrees', '${outputs}/filelist_test_octrees.txt', '${outputs}/test_octrees.tfrecords', file_type='data', shuffle=False)"""
python data_parsing.py """TFRecordsConverter.write_records('${outputs}/shape.octrees', '${outputs}/filelist_train_octrees.txt', '${outputs}/train_octrees.tfrecords', file_type='data', shuffle=False)"""
