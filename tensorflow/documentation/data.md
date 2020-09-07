#### ModelNet40
##### octree generation from .points using:
    --depth 5 --adaptive 0 --node_dis 0 --axis z


#### How many samples:
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


### Generate segmenation building data:

Look at Makefile!


given:
/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/w_colour_norm_w_labels_sample
that contains 16 files

we run 
1. python seg_buildings_no_colour.py """from_colored_annotated_data_to_default_ply('${root_dir_sample}', 'w_colour_norm_w_labels', sample_pts=1000, parallel=True)"""
which creates a ply1000 folder with 16 ply objects that have 1000 points and look like the ones created in partnet
2. python seg_buildings_no_colour.py """convert_points('${root_dir_sample}', sample_pts=1000, parallel=False)"""
which creates a points1000 folder with 16 points objects that have 1000 points and should look like the ones created in partnet
3. python seg_buildings_no_colour.py """convert_points_to_tfrecords('${root_dir_sample}', 'dataset_points_sample', sample_pts=1000)"""
which calls
python /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/data/../util/convert_tfrecords.py --file_dir /media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/points1000 --list_file /media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset_points_sample/train_points.txt --records_name /media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset_points_sample/train_points1000.tfrecords --shuffle true
