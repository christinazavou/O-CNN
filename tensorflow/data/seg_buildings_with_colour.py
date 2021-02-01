import os
import numpy as np
import sys

sys.path.append('..')
from libs import *
import tensorflow as tf
import time
import json
from tqdm import tqdm


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


def check_records():
    records_name = '/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/test_w_colour_no_rot.tfrecords'
    c = 0
    for record in tf.io.tf_record_iterator(records_name):
        c += 1
        parsed_record = tf.parse_single_example(
            record,
            {'filename': tf.FixedLenFeature([], tf.string), 'data': tf.FixedLenFeature([], tf.string)}
        )
        result = tf.Session().run(parsed_record)
        continue
        if c == 900:
            parsed_record = tf.parse_single_example(
                record,
                {'filename': tf.FixedLenFeature([], tf.string)}
            )
            filename = tf.Session().run(parsed_record)['filename']
            # assert "RESIDENTIALhouse_mesh2627_w_label" in filename.decode('utf8')

    # assert c == 1600


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


def create_points(files, w_color, label_dir, colour_dir, sess):
    byte_points_tensors = []
    for file in files:
        a = np.loadtxt(file).astype(np.float32)
        i = 0
        points = a[:, i:i + 3]  # x,y,z
        i += 3
        normals = a[:, i:i + 3]  # nx,ny,nz
        i += 3
        if w_color:
            if colour_dir:
                try:
                    c_file = os.path.basename(file)
                    lines = open(os.path.join(colour_dir, c_file[:c_file.rfind("_")] + ".ply"), "r").readlines()[14:]
                    features = np.array([line.strip().split()[-4:] for line in lines], dtype=float)
                except FileNotFoundError:
                    print("Colour file {} not found".format(
                        os.path.join(colour_dir, c_file[:c_file.rfind("_")] + ".ply")))
                    exit()
            else:
                features = a[:, i:i + 4]  # r,g,b,a
                i += 4
        else:
            features = 0.0
        if label_dir:
            try:
                l_file = os.path.basename(file)
                l = json.load(open(os.path.join(label_dir, l_file[0: l_file.rfind("_")] + "_label.json")))
                labels = np.zeros((len(l), 1))
                for k, v in l.items():
                    labels[int(k)] = v
            except FileNotFoundError:
                print("Label file {} not found".format(os.path.join(label_dir, os.path.basename(file) + "_label.json")))
                exit()
        else:
            labels = a[:, i]  # int from 0 to 33
        pts_tf = points_new(points, normals, features, labels)
        # for i in range(4,9):
        #     t = time.time()
        #     pts_tf = points_new(points, normals, features, labels)
        #     # sess.run(pts_tf)
        #     p_new = int(round((time.time() - t) * 1000))
        #     print(pts_tf)
        #     print("Points new: " + str(p_new) + " ms")
        #     t=time.time()
        #     x=custom_points2octree(in_points=pts_tf, depth=i)#, full_depth=2,
        #                            #node_dis=True, node_feature=False,split_label=False, adaptive=False, adp_depth=4,
        #                            #th_normal=0.1, save_pts=False)
        #     sess.run(x)
        #     print("CustomPoints2octree depth {} : {} ms".format(i, int(round((time.time() - t) * 1000))))
        #     print(x)
        #
        #     t=time.time()
        #     y=custom_transform_points(pts_tf,sigma=0.01,clip=0.05,angle=0.0)
        #     #sess.run(y)
        #     p_new = int(round((time.time() - t) * 1000))
        #     print(y)
        #     print("Trans points: " + str(p_new) + " ms")
        #     t=time.time()
        #     z=points2octree(in_points=y, depth=i, full_depth=2, node_dis=True, node_feature=False,
        #                           split_label=False, adaptive=False, adp_depth=4, th_normal=0.1, save_pts=False)
        #     sess.run(z)
        #     print("Points2octree depth {} : {} ms".format(i,int(round((time.time()-t) * 1000))+p_new))
        #     print(z)
        byte_points_tensors.append(pts_tf)

    return sess.run(byte_points_tensors)


def multi_chunks(lsts, n):
    lengths = [len(lst) for lst in lsts]
    assert len(set(lengths)) == 1
    for i in range(0, len(lsts[0]), n):
        yield (lst[i:i + n] for lst in lsts)


def get_data_label_index(list_file):
    file_list = []
    label_list = []
    index_list = []
    with open(list_file) as f:
        for i, line in enumerate(f):
            file, label, index = line.split()
            file_list.append(file)
            label_list.append(int(label))
            index_list.append(int(index))
    return file_list, label_list, index_list


def write_data_to_tfrecords(file_dir, list_file, records_name, file_type, label_dir=None, colour_dir=None,
                            w_color=False):
    filenames, label, index = get_data_label_index(list_file)  # label of entire model, used for classification only
    s_time = time.time()
    write_records(file_dir, records_name, file_type, filenames, label, index, label_dir, colour_dir, 8, w_color)
    e_time = time.time()
    print("Time {} m".format((e_time - s_time) // 60))
    print("Time {} ms: ".format(int(round((e_time - s_time) * 1000))))


def write_records(file_dir, records_name, file_type, filenames, label, index, label_dir, colour_dir, chunk_size=8,
                  w_color=False):
    with tf.Session() as sess:
        with tf.io.TFRecordWriter(records_name) as writer:
            chunk_data = 0
            for f_chunk, l_chunk, i_chunk in tqdm(multi_chunks([filenames, label, index], chunk_size)):
                f_chunk = [os.path.join(file_dir, filename) for filename in f_chunk]
                points_bytes = create_points(f_chunk, w_color, label_dir, colour_dir, sess)
                chunk_data += chunk_size
                if not chunk_data % (chunk_size * 10):
                    print('data loaded: {}'.format(chunk_data))
                chunk_idx = 0
                for f, l, i in zip(f_chunk, l_chunk, i_chunk):
                    feature = {
                        file_type: _bytes_feature(points_bytes[chunk_idx][0]),
                        'label': _int64_feature(l),
                        'index': _int64_feature(i),
                        'filename': _bytes_feature(('%06d_%s' % (i, f)).encode('utf8'))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    chunk_idx += 1


def split_data_label_indices_in_files(list_file, shuffle_data, count=-1, start_from=0, rot_num=0):
    file_list = []
    label_list = []
    with open(list_file) as f:
        for i, line in enumerate(f):
            if i < start_from:
                continue
            if count != -1 and i >= count + start_from:
                break
            file, label = line.split()
            if rot_num > 0:
                label_list.extend([int(label)] * rot_num)
                name, ext = file.split(".")
                file_list.extend([name + "_{:03d}.".format(i) + ext for i in range(rot_num)])
            else:
                file_list.append(file)
                label_list.append(int(label))
    if count != -1:
        assert len(label_list) == count, "{} lines read != {}".format(len(label_list), count)
    index_list = list(range(len(label_list)))

    shuffled_file = list_file.replace(".txt", "") + '.shuffle.txt'
    if start_from != 0 or count != -1:
        shuffled_file = list_file.replace(".txt", "") + 's{}c{}.shuffle.txt'.format(start_from, count)

    if shuffle_data:
        print("shuffling data")
        c = list(zip(file_list, label_list, index_list))
        shuffle(c)
        file_list, label_list, index_list = zip(*c)

    def chunk_file_name(fname, chunk):
        return "{}.{}.txt".format(fname.replace(".txt", ""), chunk)

    chunk_idx = 0
    for filenames, labels, indices in multi_chunks([file_list, label_list, index_list], 500):
        txt_file = chunk_file_name(list_file, chunk_idx) if not shuffle_data else \
            chunk_file_name(shuffled_file, chunk_idx)
        with open(txt_file, 'w') as out_file:
            for f, l, i in zip(filenames, labels, indices):
                out_file.write('{} {} {}\n'.format(f, l, i))
        chunk_idx += 1
    print("Done.")


if __name__ == '__main__':
    eval(sys.argv[1])
