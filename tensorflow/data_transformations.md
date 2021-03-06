#### ModelNet40dupl (airplane, bathtub, bed)
##### octree generation from .points:
    --depth 5 --adaptive 0 --node_dis 0 --axis z


#### calling: ```python ../data/cls_modelnet.py --run m40_generate_octree_tfrecords```

calls:

    octree --filenames /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/ModelNet40.points/xbox/test/list.txt --output_path /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/ModelNet40.octree.5/xbox/test --depth 5 --adaptive 0 --node_dis 0 --axis z

outputs:

    Processing: xbox_0110.upgrade.smp
    Processing: xbox_0119.upgrade.smp
    Processing: xbox_0122.upgrade.smp
    Done: /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/ModelNet40.points/xbox/test/list.txt
    .
    .
    .
    python /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/util/convert_tfrecords.py  --file_dir /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/ModelNet40.octree.5 --list_file /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/m40_test_octree_list.txt --records_name /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/m40_5_2_12_test_octree.tfrecords

#### calling:  ```python ../data/cls_modelnet.py --run m40_generate_points_tfrecords```
does

```1. python /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/util/convert_tfrecords.py --shuffle true --file_dir /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/ModelNet40.points --list_file /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/m40_train_points_list.txt --records_name /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/m40_train_points.tfrecords```

```2.python /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/util/convert_tfrecords.py  --file_dir /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/ModelNet40.points --list_file /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/m40_test_points_list.txt --records_name /media/christina/Data/ANFASS_data/O-CNN/ModelNet40Samples3/m40_test_points.tfrecords```


#### How many samples:
##### ModelNet40Samples3 
    120 training samples
    120 test samples.
##### ModelNetOnly4
    1247 training samples
    250 test samples.


note:
if we have 9 data .. i.e. 3 categories of 3 samples each ... then generate_point_tfrecords generates 9 tfrecords but generate_octree_tfrecords generates 9x12=108 tfrecords!

basically in the generation of octrees (with default --rot_num 12)... from one point cloud we generate 12 octrees .. and the files are named with suffix 0 to 12


Convolutions:
---
feature_reshaped: (1, 3, 119224, 1)
conv_depth5: (1, 16, 119224, 1)
pool_depth5: (1, 16, 27872, 1)
conv_depth4: (1, 32, 27872, 1)
pool_depth4: (1, 32, 6904, 1)
conv_depth3: (1, 64, 6904, 1)
pool_depth3: (1, 64, 2048, 1)

feature_reshaped: (1, 3, 117856, 1)
conv_depth5: (1, 16, 117856, 1)
pool_depth5: (1, 16, 27696, 1)
conv_depth4: (1, 32, 27696, 1)
pool_depth4: (1, 32, 7424, 1)
conv_depth3: (1, 64, 7424, 1)
pool_depth3: (1, 64, 2048, 1)


Translating shape_completion dataset from points to octrees to run autoencoder with ocnn:
 - use data_parsing.TFRecordsConverter.write_records
    - test_scans_octrees: 14400 data samples
    - test_octrees: 64608 data samples
    - train_octrees: 307080 data samples
