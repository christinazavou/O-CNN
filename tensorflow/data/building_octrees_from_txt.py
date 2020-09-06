import os
import numpy as np
import sys
sys.path.append('..')
from libs import *
import tensorflow as tf

# help(points_new)
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
