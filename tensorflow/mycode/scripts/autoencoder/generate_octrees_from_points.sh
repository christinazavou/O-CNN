#!/bin/bash

cd ../../src

python data_parsing.py """FileManipulator.generate_list_text_files('/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/shape.points')"""

python data_parsing.py """FileManipulator.generate_octrees_for_each_folder('/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/shape.points', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/shape.octrees', '--depth 6 --split_label 1 --rot_num 6')"""

python data_parsing.py """FileManipulator.point_list_to_octree_list('/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/filelist_test.txt', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/filelist_test_octrees.txt', 6, 6)"""
python data_parsing.py """FileManipulator.point_list_to_octree_list('/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/filelist_test_scans.txt', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/filelist_test_scans_octrees.txt', 6, 6)"""
python data_parsing.py """FileManipulator.point_list_to_octree_list('/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/filelist_train.txt', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/filelist_train_octrees.txt', 6, 6)"""

python data_parsing.py """TFRecordsConverter.write_records('/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/shape.octrees', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/filelist_test_octrees.txt', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_octrees.tfrecords', file_type='data', shuffle=False)"""
# loaded 32304 data samples
python data_parsing.py """TFRecordsConverter.write_records('/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/shape.octrees', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/filelist_test_scans_octrees.txt', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_scans_octrees.tfrecords', file_type='data', shuffle=False)"""
# loaded 7200 data samples
python data_parsing.py """TFRecordsConverter.write_records('/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/shape.octrees', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/filelist_train_octrees.txt', '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_train_octrees.tfrecords', file_type='data', shuffle=False)"""
# loaded 153540 data samples
