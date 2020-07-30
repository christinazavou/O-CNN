import sys

import numpy as np

sys.path.append("..")
from data_parsing import *


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


class DatasetTest(tf.test.TestCase):

    # def test_data(self):
    #     flags = make_flags()
    #     octree, label = DatasetFactory(flags.DATA.train)()
    #     print("octree: ", octree)
    #     with self.cached_session(use_gpu=True):
    #         octree_1 = octree.eval()
    #         print(octree_1, len(octree_1))
    #     self.assertTrue(octree is not None)
    def test_all(self):
        with self.cached_session(use_gpu=True) as sess:
            self.test_point_dataset(sess)
            self.test_point_cloud_dataset(sess)

    def test_point_dataset(self, sess):
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
        self.assertTrue(np.issubdtype(merged_octrees_batch1[0].dtype, np.integer))
        self.assertTrue(np.issubdtype(merged_octrees_batch1[1].dtype, np.integer))
        self.assertEqual(merged_octrees_batch1[1].shape, (flags.batch_size,))

        merged_octrees_batch2 = sess.run(
            points(tf_record_filenames=flags.location,
                   batch_size=flags.batch_size,
                   shuffle_size=flags.shuffle,
                   return_iterator=flags.return_iterator,
                   take=flags.n_samples,
                   return_pts=flags.return_pts))
        self.assertTrue(np.issubdtype(merged_octrees_batch2[0].dtype, np.integer))
        self.assertTrue(np.issubdtype(merged_octrees_batch2[1].dtype, np.integer))
        self.assertEqual(merged_octrees_batch2[1].shape, (flags.batch_size,))

        self.assertTrue(merged_octrees_batch2[0].shape != merged_octrees_batch1[0].shape)
        print("check 1")

    def test_point_cloud_dataset(self, sess):
        points = PointCloudDataset(ParseExample())

        next_point = sess.run(
            points(
                tf_record_filenames='/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/dataset/ocnn_completion/completion_test_points.tfrecords',
                batch_size=10,
                shuffle_size=1000,
                return_iterator=False,
                take=-1))
        self.assertEqual(next_point[0].shape, (10,))  # batch point clouds
        self.assertEqual(next_point[0].dtype, object)  # point cloud represented as string
        self.assertEqual(next_point[1].shape, (10,))  # batch point cloud labels
        self.assertEqual(next_point[1].dtype, int)  # point cloud label represented as int
        print("check 2")

    # def test_point_dataset(self):
    #     flags = make_flags().DATA.train
    #
    #     with tf.Session() as tmpSess:
    #         octree_batch = tmpSess.run(octree_batch(octree_samples(['octree_1', 'octree_2'])))
    #         print("octree_batch", octree_batch)


if __name__ == "__main__":
    tf.test.main()
