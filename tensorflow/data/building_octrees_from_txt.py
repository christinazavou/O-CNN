import os
import numpy as np
import sys
sys.path.append('..')
from libs import *
import tensorflow as tf

# help(points_new)

root = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/w_colour_norm_w_labels'
for filename in os.listdir(root):
    filepath = os.path.join(root, filename)
    print(filepath)
    a = np.loadtxt(filepath).astype(np.float32)
    points = a[:, 0:3]
    normals = a[:, 3:6]
    features = a[:, 6:10]
    labels = a[:, 10]#.astype(np.int)
    pts_tf = points_new(points, normals, features, labels)

    pts_tf_label = points_property(pts_tf, property_name='label', channel=1)
    pts_tf_xyz = points_property(pts_tf, property_name='xyz', channel=3)
    pts_tf_xyzd = points_property(pts_tf, property_name='xyz', channel=4)
    pts_tf_feature = points_property(pts_tf, property_name='feature', channel=4)
    with tf.Session() as sess:
        assert np.array_equal(sess.run(pts_tf_label), labels.reshape((-1, 1)))
        assert np.array_equal(sess.run(pts_tf_xyz), points)
        assert np.array_equal(sess.run(pts_tf_xyzd)[:, 3], np.zeros(len(points)))
        assert np.array_equal(sess.run(pts_tf_feature), features)

    break
