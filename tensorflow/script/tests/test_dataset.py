import sys
sys.path.append("../../libs")
from unittest.mock import patch
import tensorflow as tf
import numpy as np

from config import parse_args
from dataset import DatasetFactory
from libs import octree_property, points_property


def test_dataset_factory():
    testargs = ["python test_config.py",
                "--config", "../configs/segmentation/seg_hrnet_partnet_pts.yaml",
                "DATA.test.shuffle", "0"]
    with patch.object(sys, 'argv', testargs):
        FLAGS = parse_args()
    dataset = DatasetFactory(FLAGS.DATA.test)(return_iter=True, return_fnames=True)
    data_tuple = dataset.get_next()
    assert len(data_tuple) == 4
    with tf.Session() as sess:
        filenames = sess.run(data_tuple[-1])
        assert filenames.shape[0] == 1
        assert "RESIDENTIALvilla_mesh5826" in str(filenames[0])


def test_dataset_factory_octree_properties():
    testargs = ["python test_config.py",
                "--config", "../configs/segmentation/seg_hrnet_partnet_pts.yaml",
                "DATA.test.shuffle", "0"]
    with patch.object(sys, 'argv', testargs):
        FLAGS = parse_args()
    dataset = DatasetFactory(FLAGS.DATA.test)(return_iter=True, return_fnames=True)
    octree, labels, points, filenames = dataset.get_next()
    with tf.Session() as sess:
        octree_features = sess.run(octree_property(octree, property_name='feature', dtype=tf.float32,
                                                   depth=FLAGS.MODEL.depth, channel=FLAGS.MODEL.channel))
        assert octree_features.shape[0] == FLAGS.MODEL.channel
        assert octree_features.shape[1] == 7280
        assert octree_features != []
        # just a check for nx,ny,nz values
        assert -1 <= np.min(octree_features[0:3, :]) <= 1 and -1 <= np.max(octree_features[0:3, :]) <= 1
        if octree_features.shape[0] > 4:
            # just a check for r,g,b,a values
            assert 0 <= np.min(octree_features[-4:, :]) <= 1 and 0 <= np.max(octree_features[-4:, :]) <= 1


def test_dataset_factory_points_properties():
    testargs = ["python test_config.py",
                "--config", "../configs/segmentation/seg_hrnet_partnet_pts.yaml",
                "DATA.test.shuffle", "0"]
    with patch.object(sys, 'argv', testargs):
        FLAGS = parse_args()
    dataset = DatasetFactory(FLAGS.DATA.test)(return_iter=True, return_fnames=True)
    octree, labels, points, filenames = dataset.get_next()
    if FLAGS.DATA.test.node_dis:
        extra_feature_channels = FLAGS.MODEL.channel - 4
    else:
        extra_feature_channels = FLAGS.MODEL.channel - 3
    with tf.Session() as sess:
        point_normals = sess.run(points_property(points, property_name='normal', channel=3))
        assert point_normals.shape[1] == 3
        point_features = sess.run(points_property(points, property_name='feature', channel=extra_feature_channels))
        assert point_features.shape[1] == extra_feature_channels
        assert point_normals.shape[0] == point_features.shape[0]
