#!/bin/bash

#eval "$(conda shell.bash hook)"
#conda activate ocnn_tf1.14

point_loc='/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/nocolor_12_rot_correct/'
label_loc='/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/labels_32'
colour_loc='/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/withcolor'
data_split='/media/maria/BigData1/Maria/caffe_ocnn_seg_tf/experiments/buildnet_data/train_points.shuffle.'
dest_loc_no='/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/tfrecords/train_no_colour_no_rot.'
dest_loc_w='/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/tfrecords/train_w_colour_no_rot.'
logfile='/media/maria/BigData1/Maria/caffe_ocnn_seg_tf/experiments/buildnet_data/train_logs.txt'

#cat ${dest_loc_no}0.tfrecords > /media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/tfrecords/train_no_colour_12_rot.shuffle.all.tfrecords
cat ${dest_loc_w}0.tfrecords > /media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/tfrecords/train_w_colour_12_rot.shuffle.all.tfrecords
#cat ${dest_loc_no}1.tfrecords >> /media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/tfrecords/train_no_colour_12_rot.shuffle.all.tfrecords
cat ${dest_loc_w}1.tfrecords >> /media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/tfrecords/train_w_colour_12_rot.shuffle.all.tfrecords

for i in `seq 2 38`;
  do
    #python seg_buildings_with_colour.py """write_data_to_tfrecords('${point_loc}', '${data_split}${i}.txt', '${dest_loc_no}${i}.tfrecords', 'data','${label_loc}','${colour_loc}',False)""" >> ${logfile}
    python seg_buildings_with_colour.py """write_data_to_tfrecords('${point_loc}', '${data_split}${i}.txt', '${dest_loc_w}${i}.tfrecords', 'data','${label_loc}','${colour_loc}',True)""" >> ${logfile}
    #cat ${dest_loc_no}${i}.tfrecords >> /media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/tfrecords/train_no_colour_12_rot.shuffle.all.tfrecords
    cat ${dest_loc_w}${i}.tfrecords >> /media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/tfrecords/train_w_colour_12_rot.shuffle.all.tfrecords
  done