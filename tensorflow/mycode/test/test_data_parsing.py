import sys
import unittest
from unittest import TestCase

import numpy as np

sys.path.append("../..")
from src.data_parsing import *


def make_flags():
    from yacs.config import CfgNode as CN

    _C = CN()

    # DATA related parameters
    _C.DATA = CN()
    _C.DATA.train = CN()
    _C.DATA.train.dtype = 'points'  # The data type: points or octree
    _C.DATA.train.x_alias = 'data'  # The alias of the data
    _C.DATA.train.y_alias = 'label'  # The alias of the target

    _C.DATA.train.depth = 5  # The octree depth
    _C.DATA.train.full_depth = 2  # The full depth
    _C.DATA.train.node_dis = False  # Save the node displacement
    _C.DATA.train.split_label = False  # Save the split label
    _C.DATA.train.adaptive = False  # Build the adaptive octree
    _C.DATA.train.node_feat = False  # Calculate the node feature

    _C.DATA.train.distort = False  # Whether to apply data augmentation
    _C.DATA.train.offset = 0.55  # Offset used to displace the points
    _C.DATA.train.axis = 'y'  # Rotation axis for data augmentation
    _C.DATA.train.scale = 0.0  # Scale the points
    _C.DATA.train.uniform = False  # Generate uniform scales
    _C.DATA.train.jitter = 0.0  # Jitter the points
    _C.DATA.train.drop_dim = (8, 32)  # The value used to dropout points
    _C.DATA.train.dropout = (0, 0)  # The dropout ratio
    _C.DATA.train.stddev = (0, 0, 0)  # The standard deviation of the random noise
    _C.DATA.train.interval = (1, 1, 1)  # Use interval&angle to generate random angle
    _C.DATA.train.angle = (180, 180, 180)

    _C.DATA.train.location = '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/dataset/ocnn_completion/completion_test_points.tfrecords'  # The data location
    _C.DATA.train.shuffle = 1000  # The shuffle size
    _C.DATA.train.n_samples = -1  # Use at most `n_samples` elements from this dataset
    _C.DATA.train.batch_size = 32  # Training data batch size
    _C.DATA.train.mask_ratio = 0.0  # Mask out some point features

    _C.DATA.train.return_iterator = False  # Return the data iterator
    _C.DATA.train.return_pts = False  # Also return points

    return _C


class Octrees2TFRecordsFileTest(TestCase):

    def test_get_data_label_pair(self):
        filepaths, labels, indices = OctreesTFRecordsConverter \
            .get_data_label_pair("resources/m40_test_points_list_sample.txt")
        self.assertTrue(len(filepaths) == len(labels) == len(indices) == 5)

    @unittest.SkipTest
    def test_write_records(self):
        OctreesTFRecordsConverter \
            .write_records("resources/ModelNet40.octree.5.sample", "resources/m40_test_points_list_sample.txt",
                           "resources/m40_test_points_sample.tfrecords", file_type='data', shuffle=False)

    def test_read_records(self):
        OctreesTFRecordsConverter \
            .read_records("/media/christina/Data/ANFASS_data/O-CNN/ModelNet40/m40_test_points.tfrecords",
                          "./resources/octrees_from_tfrecords",
                          "/media/christina/Data/ANFASS_data/O-CNN/ModelNet40/m40_test_octree_list.txt",
                          file_type='data', count=5)


class DatasetTest(tf.test.TestCase):

    def setUp(self):
        self.verificationErrors = []

    def tearDown(self):
        self.assertEqual([], self.verificationErrors)

    def test_point_dataset(self):
        with tf.Session() as sess:
            flags = make_flags().DATA.train
            points = PointDataset(ParseExample(**flags),
                                  TransformPoints(**flags, bounding_sphere=bounding_sphere),
                                  Points2Octree(**flags))

            merged_octrees_batch1 = sess.run(
                points(tf_record_filenames=flags.location,
                       batch_size=flags.batch_size,
                       shuffle_size=flags.shuffle,
                       return_iterator=flags.return_iterator,
                       take=flags.n_samples,
                       return_pts=flags.return_pts))

            try:
                self.assertTrue(np.issubdtype(merged_octrees_batch1[0].dtype, np.integer))
                self.assertTrue(np.issubdtype(merged_octrees_batch1[1].dtype, np.integer))
                self.assertEqual(merged_octrees_batch1[1].shape, (flags.batch_size,))
            except AssertionError as e:
                self.verificationErrors.append(str(e))

            merged_octrees_batch2 = sess.run(
                points(tf_record_filenames=flags.location,
                       batch_size=flags.batch_size,
                       shuffle_size=flags.shuffle,
                       return_iterator=flags.return_iterator,
                       take=flags.n_samples,
                       return_pts=flags.return_pts))

            try:
                self.assertTrue(np.issubdtype(merged_octrees_batch2[0].dtype, np.integer))
                self.assertTrue(np.issubdtype(merged_octrees_batch2[1].dtype, np.integer))
                self.assertEqual(merged_octrees_batch2[1].shape, (flags.batch_size,))

                self.assertTrue(merged_octrees_batch2[0].shape != merged_octrees_batch1[0].shape)

            except AssertionError as e:
                self.verificationErrors.append(str(e))
        print("test_point_dataset checked")

    def test_octree_dataset(self):
        with tf.Session() as sess:
            flags = make_flags().DATA.train
            octrees = OctreeDataset(ParseExample(**flags))

            merged_octrees_batch1 = sess.run(
                octrees(
                    tf_record_filenames='/media/christina/Data/ANFASS_data/O-CNN/ModelNet40/m40_5_2_12_test_octree.tfrecords',
                    batch_size=32,
                    shuffle_size=False,
                    return_iterator=False,
                    take=10))
            print(merged_octrees_batch1)
            try:
                self.assertTrue(np.issubdtype(merged_octrees_batch1[0].dtype, np.integer))
                self.assertTrue(np.issubdtype(merged_octrees_batch1[1].dtype, np.integer))
                self.assertEqual(merged_octrees_batch1[1].shape, (flags.batch_size,))
            except AssertionError as e:
                self.verificationErrors.append(str(e))

        print("test_octree_dataset checked")

    def test_point_cloud_dataset(self):
        with tf.Session() as sess:
            next_point = PointCloudDataset(ParseExample())

            point_cloud_1 = sess.run(
                next_point(
                    tf_record_filenames='/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/dataset/ocnn_completion/completion_test_points.tfrecords',
                    batch_size=10,
                    shuffle_size=1000,
                    return_iterator=False,
                    take=-1))

            point_cloud_2 = sess.run(
                next_point(
                    tf_record_filenames='/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/dataset/ocnn_completion/completion_test_points.tfrecords',
                    batch_size=10,
                    shuffle_size=1000,
                    return_iterator=False,
                    take=-1))

            try:
                self.assertEqual(point_cloud_1[0].shape, (10,))  # batch point clouds
                self.assertEqual(point_cloud_1[0].dtype, object)  # point cloud represented as string
                self.assertEqual(point_cloud_1[1].shape, (10,))  # batch point cloud labels
                self.assertEqual(point_cloud_1[1].dtype, int)  # point cloud label represented as int

                # NOTE: becuase next_point is calling the next element of the generated TFRecordDataset everytime we call it we get the next element ;) !
                self.assertFalse(all(point_cloud_1[0] == point_cloud_2[0]))
                self.assertFalse(all(point_cloud_1[1] == point_cloud_2[1]))
            except AssertionError as e:
                self.verificationErrors.append(str(e))

        print("test_point_cloud_dataset checked")


if __name__ == "__main__":
    tf.test.main()
