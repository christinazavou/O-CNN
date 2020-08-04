import sys
import unittest
from unittest import TestCase

import numpy as np

sys.path.append("../..")
from src.data_parsing import *


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
            points = PointDataset(ParseExample(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=5, offset=0.55, axis='y', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180], bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=5))

            merged_octrees_batch1 = sess.run(
                points(
                    tf_record_filenames='/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/dataset/ocnn_completion/completion_test_points.tfrecords',
                    batch_size=32,
                    shuffle_size=1000,
                    return_iterator=False,
                    take=-1,
                    return_pts=False))

            try:
                self.assertTrue(np.issubdtype(merged_octrees_batch1[0].dtype, np.integer))
                self.assertTrue(np.issubdtype(merged_octrees_batch1[1].dtype, np.integer))
                self.assertEqual(merged_octrees_batch1[1].shape, (32,))
            except AssertionError as e:
                self.verificationErrors.append(str(e))

            merged_octrees_batch2 = sess.run(
                points(
                    tf_record_filenames='/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/dataset/ocnn_completion/completion_test_points.tfrecords',
                    batch_size=32,
                    shuffle_size=1000,
                    return_iterator=False,
                    take=-1,
                    return_pts=False))

            try:
                self.assertTrue(np.issubdtype(merged_octrees_batch2[0].dtype, np.integer))
                self.assertTrue(np.issubdtype(merged_octrees_batch2[1].dtype, np.integer))
                self.assertEqual(merged_octrees_batch2[1].shape, (32,))

                self.assertTrue(merged_octrees_batch2[0].shape != merged_octrees_batch1[0].shape)

            except AssertionError as e:
                self.verificationErrors.append(str(e))
        print("test_point_dataset checked")

    def test_octree_dataset(self):
        with tf.Session() as sess:
            octrees = OctreeDataset(ParseExample(x_alias='data', y_alias='label'))

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
                self.assertEqual(merged_octrees_batch1[1].shape, (32,))
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
