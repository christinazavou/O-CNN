#!/bin/bash

#eval "$(conda shell.bash hook)"
#conda activate ocnn_tf1.14

point_loc='/media/maria/BigData1/Maria/nocolor_12_rot_correct/'
label_loc='/media/yiangos/BigData/Maria/point_labels_32'
colour_loc='/media/yiangos/BigData/Maria/withcolor'
data_split='/media/yiangos/BigData/Maria/tfrecord_splits/'
dest_loc='/media/maria/BigData1/Maria/tfrecords/'
logfile='/media/yiangos/BigData/Maria/tfrecords_logs.txt'

python seg_buildings_with_colour.py """write_data_to_tfrecords('${point_loc}', '${data_split}test_points.0.txt', '${dest_loc}test_w_colour_no_rot.tfrecords', 'data','${label_loc}','${colour_loc}',True)""" >> ${logfile}
python seg_buildings_with_colour.py """write_data_to_tfrecords('${point_loc}', '${data_split}val_points.shuffle.0.txt', '${dest_loc}val_w_colour_no_rot.tfrecords', 'data','${label_loc}','${colour_loc}',True)""" >> ${logfile}
python seg_buildings_with_colour.py """write_data_to_tfrecords('${point_loc}', '${data_split}train_points.shuffle.0.txt', '${dest_loc}train_w_colour_12_rot.0.tfrecords', 'data','${label_loc}','${colour_loc}',True)""" >> ${logfile}

cat ${dest_loc}train_w_colour_12_rot.0.tfrecords > ${dest_loc}train_w_colour_12_rot.shuffle.all.tfrecords

for i in `seq 1 38`;
  do
    python seg_buildings_with_colour.py """write_data_to_tfrecords('${point_loc}', '${data_split}train_points.shuffle.${i}.txt', '${dest_loc}train_w_colour_12_rot.${i}.tfrecords', 'data','${label_loc}','${colour_loc}',True)""" >> ${logfile}
	cat ${dest_loc}train_w_colour_12_rot.${i}.tfrecords >> ${dest_loc}train_w_colour_12_rot.shuffle.all.tfrecords      
  done

echo "Done."