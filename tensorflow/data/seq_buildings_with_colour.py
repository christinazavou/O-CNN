import os
import numpy as np
import sys
sys.path.append('..')
from libs import *
import tensorflow as tf
# help(points_new)

def debug():
    depth = 6

    root = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/w_colour_norm_w_labels'
    for filename in os.listdir(root):
        filepath = os.path.join(root, filename)
        print(filepath)
        a = np.loadtxt(filepath).astype(np.float32)
        points = a[:, 0:3]  # x,y,z
        normals = a[:, 3:6]  # nx,ny,nz
        features = a[:, 6:10]  # r,g,b,a
        labels = a[:, 10]  # int from 0 to 33

        pts_tf = points_new(points, normals, features, labels)

        pts_tf_label = points_property(pts_tf, property_name='label', channel=1)
        pts_tf_xyz = points_property(pts_tf, property_name='xyz', channel=3)
        pts_tf_xyzd = points_property(pts_tf, property_name='xyz', channel=4)
        pts_tf_feature = points_property(pts_tf, property_name='feature', channel=4)

        octree_tf = points2octree(pts_tf, depth=depth, full_depth=2, node_dis=True, node_feature=False,
                                  split_label=False, adaptive=False, adp_depth=4, th_normal=0.1, save_pts=True)
        octree = octree_batch([octree_tf])

        # FIXME: TODO: ask whether is correct to have x,y,z in the input and why they dont use them in default(partnet) where we had 4 feature channels?
        # feature: average x,y,z,nx,ny,nz,dis,r,g,b,a at max depth
        octree_feature_tf = octree_property(octree, property_name='feature', depth=depth, dtype=tf.float32, channel=11)
        octree_label_tf = octree_property(octree, property_name='label', depth=depth, dtype=tf.float32, channel=1)
        octree_xyz_tf = octree_property(octree, property_name='xyz', depth=depth, dtype=tf.uint32, channel=1)
        octree_index_tf = octree_property(octree, property_name='index', depth=depth, dtype=tf.int32, channel=1)

        with tf.Session() as sess:
            assert np.array_equal(sess.run(pts_tf_label), labels.reshape((-1, 1)))
            assert np.array_equal(sess.run(pts_tf_xyz), points)
            assert np.array_equal(sess.run(pts_tf_xyzd)[:, 3], np.zeros(len(points)))
            assert np.array_equal(sess.run(pts_tf_feature), features)

            pts_tf_val = sess.run(pts_tf)  # binary
            assert type(pts_tf_val[0]) == bytes
            octree_val = sess.run(octree_tf)  # binary
            assert type(octree_val[0]) == bytes

            octree_val = sess.run(octree)
            assert len(octree_val.shape) == 1

            o_feature, o_label, o_xyz, o_idx = sess.run([octree_feature_tf, octree_label_tf, octree_xyz_tf,
                                                         octree_index_tf])
            print("done")

        break


import os
import argparse
from random import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--file_dir', type=str, required=True,
                    help='Base folder containing the data')
parser.add_argument('--list_file', type=str, required=True,
                    help='File containing the list of data')
parser.add_argument('--records_name', type=str, required=True,
                    help='Name of tfrecords')
parser.add_argument('--file_type', type=str, required=False, default='data',
                    help='File type')
parser.add_argument('--shuffle_data', type=bool, required=False, default=False,
                    help='Whether to shuffle the data order')
parser.add_argument('--count', type=int, required=False, default=-1,
                    help='Amount of cases to include')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_points(file, sess):
    a = np.loadtxt(file).astype(np.float32)
    points = a[:, 0:3]  # x,y,z
    normals = a[:, 3:6]  # nx,ny,nz
    features = a[:, 6:10]  # r,g,b,a
    labels = a[:, 10]  # int from 0 to 33

    pts_tf = points_new(points, normals, features, labels)
    return sess.run(pts_tf[0])


def write_data_to_tfrecords(file_dir, list_file, records_name, file_type, shuffle_data, count):
    [data, label, index] = get_data_label_pair(list_file, shuffle_data, count)

    writer = tf.python_io.TFRecordWriter(records_name)
    with tf.Session() as sess:
        for i in range(len(data)):
            if not i % 1000:
                print('data loaded: {}/{}'.format(i, len(data)))

            points_bytes = create_points(os.path.join(file_dir, data[i]), sess)
            feature = {file_type: _bytes_feature(points_bytes),
                       'label': _int64_feature(label[i]),
                       'index': _int64_feature(index[i]),
                       'filename': _bytes_feature(('%06d_%s' % (i, data[i])).encode('utf8'))}
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
    writer.close()


def get_data_label_pair(list_file, shuffle_data, count):
    file_list = []
    label_list = []
    with open(list_file) as f:
        for i, line in enumerate(f):
            if count != -1 and i > count:
                break
            file, label = line.split()
            file_list.append(file)
            label_list.append(int(label))
    index_list = list(range(len(label_list)))

    if shuffle_data:
        print("shuffling data")
        c = list(zip(file_list, label_list, index_list))
        shuffle(c)
        file_list, label_list, index_list = zip(*c)
        with open(list_file + '.shuffle.txt', 'w') as f:
            for item in c:
                f.write('{} {}\n'.format(item[0], item[1]))
    return file_list, label_list, index_list


if __name__ == '__main__':
    # args = parser.parse_args()
    # write_data_to_tfrecords(args.file_dir,
    #                         args.list_file,
    #                         args.records_name,
    #                         args.file_type,
    #                         args.shuffle_data,
    #                         args.count)
    #
    prefix = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour_sample'
    # write_data_to_tfrecords(prefix+'/w_colour_norm_w_labels',
    #                         prefix+'/dataset_points_sample/train.txt',
    #                         prefix+'/dataset_points_sample/train.tfrecords',
    #                         'data',
    #                         True,
    #                         -1)
    write_data_to_tfrecords(prefix+'/w_colour_norm_w_labels',
                            prefix+'/dataset_points_sample/test.txt',
                            prefix+'/dataset_points_sample/test.tfrecords',
                            'data',
                            True,
                            -1)