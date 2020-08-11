import sys
from unittest import TestCase

import numpy as np

sys.path.append("../..")
from src.data_parsing import *


class ClassificationDatasetStatisticsTest(TestCase):

    def test_all(self):
        cds = ClassificationDatasetStatistics("/media/christina/Data/ANFASS_data/O-CNN/ModelNet40/ModelNet40.points")
        # train_samples, test_samples, categories = zip(
        #     *sorted(zip(cds.train_samples, cds.test_samples, cds.categories), reverse=True))
        # Visualizer.bar_plot(train_samples, test_samples, categories, 'Train samples', 'Test samples')
        self.assertEqual(len(cds.categories), 40)
        self.assertAlmostEqual(cds.train_n + cds.test_n, 12311, delta=5)
        print("Test data:", cds.test_n)
        print("Training data:", cds.train_n)


class TFRecordsConverterTest(TestCase):

    def test_get_data_label_pair(self):
        filepaths, labels, indices = TFRecordsConverter \
            .get_data_label_pair("resources/ModelNetOnly4Samples3/m40_test_points_list.txt")
        self.assertTrue(len(filepaths) == len(labels) == len(indices) == 9)

    def test_write_records(self):
        TFRecordsConverter \
            .write_records("resources/ModelNetOnly4Samples3/ModelNet40.octree.5",
                           "resources/ModelNetOnly4Samples3/m40_test_octree_list.txt",
                           "output/ModelNetOnly4Samples3/test_write_records/m40_5_2_12_test_octree.tfrecords",
                           file_type='data', shuffle=False)
        self.assertTrue(os.stat('output/ModelNetOnly4Samples3/test_write_records/m40_5_2_12_test_octree.tfrecords')
                        .st_size >= 8.1 * 1e6)
        TFRecordsConverter \
            .write_records("resources/ModelNetOnly4Samples3/ModelNet40.points",
                           "resources/ModelNetOnly4Samples3/m40_test_points_list.txt",
                           "output/ModelNetOnly4Samples3/test_write_records/m40_test_points.tfrecords",
                           file_type='data', shuffle=False)
        self.assertTrue(os.stat('output/ModelNetOnly4Samples3/test_write_records/m40_test_points.tfrecords')
                        .st_size >= 1.3 * 1e6)

    def test_read_records(self):
        TFRecordsConverter \
            .read_records("resources/ModelNetOnly4Samples3/m40_5_2_12_test_octree.tfrecords",
                          "output/ModelNetOnly4Samples3/test_read_records/ModelNet40.octree.5",
                          "m40_test_octree_list.txt",
                          file_type='data', count=0)
        self.assertEqual(len(os.listdir("output/ModelNetOnly4Samples3/test_read_records/ModelNet40.octree.5")),
                         108 + 1)
        TFRecordsConverter \
            .read_records("resources/ModelNetOnly4Samples3/m40_test_points.tfrecords",
                          "output/ModelNetOnly4Samples3/test_read_records/ModelNet40.points",
                          "m40_test_points_list.txt",
                          file_type='data', count=0)
        self.assertEqual(len(os.listdir("output/ModelNetOnly4Samples3/test_read_records/ModelNet40.points")),
                         9 + 1)


class DatasetTest(tf.test.TestCase):

    def setUp(self):
        self.verificationErrors = []
        self.points = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                          TransformPoints(distort=False, depth=5, offset=0.55, axis='z', scale=0.0,
                                                          jitter=0.0, angle=[180, 180, 180],
                                                          bounding_sphere=bounding_sphere),
                                          Points2Octree(depth=5))
        self.octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))

    def tearDown(self):
        self.assertEqual([], self.verificationErrors)

    def test_point_dataset(self):

        with tf.Session() as sess:
            for filename in ['resources/ModelNetOnly4Samples3/m40_test_points.tfrecords',
                             'resources/ocnn_completion_only2samples2/completion_test_points.tfrecords']:

                merged_octrees_batch1 = sess.run(self.points(tf_record_filenames=filename, batch_size=32,
                                                             shuffle_size=1000, return_iterator=False, take=-1,
                                                             return_pts=False))
                merged_octrees_batch2 = sess.run(self.points(tf_record_filenames=filename, batch_size=32,
                                                             shuffle_size=1000, return_iterator=False, take=-1,
                                                             return_pts=False))
                merged_octrees_batch3 = sess.run(self.points(tf_record_filenames=filename, batch_size=32,
                                                             shuffle_size=1000, return_iterator=False, take=-1,
                                                             return_pts=True))

                try:
                    self.assertTrue(np.issubdtype(merged_octrees_batch1[0].dtype, np.integer))
                    self.assertTrue(np.issubdtype(merged_octrees_batch1[1].dtype, np.integer))
                    self.assertEqual(merged_octrees_batch1[1].shape, (32,))

                    self.assertTrue(np.issubdtype(merged_octrees_batch2[0].dtype, np.integer))
                    self.assertTrue(np.issubdtype(merged_octrees_batch2[1].dtype, np.integer))
                    self.assertEqual(merged_octrees_batch2[1].shape, (32,))

                    self.assertTrue(merged_octrees_batch2[0].shape != merged_octrees_batch1[0].shape)
                    self.assertTrue(len(merged_octrees_batch1) == 2)
                    self.assertTrue(len(merged_octrees_batch2) == 2)
                    self.assertTrue(len(merged_octrees_batch3) == 3)
                except AssertionError as e:
                    self.verificationErrors.append(str(e))
        print("test_point_dataset checked")

    def test_octree_dataset(self):
        with tf.Session() as sess:
            call_octrees = \
                self.octrees(tf_record_filenames='resources/ModelNetOnly4Samples3/m40_5_2_12_test_octree.tfrecords',
                             batch_size=32, shuffle_size=False, return_iterator=False, take=10)

            octrees, labels, filenames = sess.run(call_octrees)
            try:
                self.assertTrue(np.issubdtype(octrees.dtype, np.integer))
                self.assertTrue(np.issubdtype(labels.dtype, np.integer))
                self.assertTrue(np.issubdtype(filenames.dtype, np.object))
                self.assertEqual(len(octrees.shape), 1)
                self.assertEqual(labels.shape, (32,))
                self.assertEqual(filenames.shape, (32,))
            except AssertionError as e:
                self.verificationErrors.append(str(e))

        print("test_octree_dataset checked")

    def test_octree_dataset_batch1(self):
        with tf.Session() as sess:
            call_octree = \
                self.octrees(tf_record_filenames='resources/ModelNetOnly4Samples3/m40_5_2_12_test_octree.tfrecords',
                             batch_size=3, shuffle_size=False, return_iterator=False, take=10)

            octree_b, labels_b, filenames_b = sess.run(call_octree)
            try:
                self.assertTrue(octree_b.shape, (3,))
                self.assertTrue('test' in str(filenames_b[0]))
            except AssertionError as e:
                self.verificationErrors.append(str(e))

        print("test_octree_dataset_batch1 checked")

    def test_point_cloud_dataset(self):
        with tf.Session() as sess:
            next_point = PointCloudDataset(ParseExampleDebug())
            call_next_point = next_point(
                tf_record_filenames='resources/ModelNetOnly4Samples3/m40_test_points.tfrecords',
                batch_size=10, shuffle_size=1000, return_iterator=False, take=-1)

            batch_point_clouds1, batch_labels1, batch_filenames1 = sess.run(call_next_point)
            batch_point_clouds2, batch_labels2, batch_filenames2 = sess.run(call_next_point)

            try:
                self.assertEqual(batch_point_clouds1.shape, (10,))  # batch point clouds
                self.assertEqual(batch_point_clouds1.dtype, object)  # point cloud represented as string
                self.assertEqual(batch_labels1.shape, (10,))  # batch point cloud labels
                self.assertEqual(batch_labels1.dtype, int)  # point cloud label represented as int

                # NOTE: because next_point is calling the next element of the generated TFRecordDataset
                # everytime we call it we get the next element ;) !
                self.assertFalse(all(batch_point_clouds1 == batch_point_clouds2))
                self.assertFalse(all(batch_labels1 == batch_labels2))
            except AssertionError as e:
                self.verificationErrors.append(str(e))

        print("test_point_cloud_dataset checked")

    @staticmethod
    def run_requirements():
        TFRecordsConverter \
            .write_records("resources/ModelNetOnly4Samples3/ModelNet40.points",
                           "resources/ModelNetOnly4Samples3/m40_test_points_list_sample1.txt",
                           "intermediate/ModelNetOnly4Samples3/test_octree_and_points"
                           "/m40_test_points_sample1.tfrecords",
                           file_type='data', shuffle=False)

        TFRecordsConverter \
            .write_records("resources/ModelNetOnly4Samples3/ModelNet40.octree.5",
                           "resources/ModelNetOnly4Samples3/m40_test_octree_list_sample1.txt",
                           "intermediate/ModelNetOnly4Samples3/test_octree_and_points"
                           "/m40_5_2_12_test_octree_sample1.tfrecords",
                           file_type='data', shuffle=False)

    def test_octree_and_points(self):
        self.run_requirements()

        with tf.Session() as sess:
            # We need a folder with:
            # 1. tfrecords file containing the bathtub_0001 points
            # 2. tfrecords file containing the bathtub_0001 octree

            call_points = self.points(tf_record_filenames='intermediate/ModelNetOnly4Samples3/test_octree_and_points'
                                                          '/m40_test_points_sample1.tfrecords',
                                      batch_size=1, shuffle_size=0, return_iterator=False, take=-1, return_pts=False)

            octree_from_points_first = sess.run(call_points)

            octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
            call_octrees = octrees(tf_record_filenames='intermediate/ModelNetOnly4Samples3/test_octree_and_points'
                                                       '/m40_5_2_12_test_octree_sample1.tfrecords',
                                   batch_size=1, shuffle_size=0, return_iterator=False, take=-1)
            octree_from_octrees_first = sess.run(call_octrees)

            found_equal = np.all(octree_from_octrees_first[0] == octree_from_points_first[0])
            print(found_equal)
            found_equal_size = octree_from_points_first[0].shape == octree_from_octrees_first[0].shape
            print(found_equal_size)

            found_equal_points = []
            for i in range(11):
                octree_from_points = sess.run(call_points)
                found_equal_points.append(np.all(octree_from_points[0] == octree_from_points_first[0]))

                octree_from_octrees = sess.run(call_octrees)
                found_equal = np.all(octree_from_octrees[0] == octree_from_points_first[0])
                print(found_equal)

            self.assertTrue(all(found_equal_points))
            print("test_octree_and_points checked")
            # ****** We have 12 octrees in the OctreeDataset and 1 points in the PointsDataset.
            # ****** Both datasets are translated into octrees though.


if __name__ == "__main__":
    tf.test.main()
