import tensorflow as tf

from data_parsing import DatasetFactory


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

    _C.DATA.train.location = '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/dataset/ModelNet40/m40_5_2_12_train_octree.tfrecords'  # The data location
    _C.DATA.train.shuffle = 1000  # The shuffle size
    _C.DATA.train.n_samples = -1  # Use at most `n_samples` elements from this dataset
    _C.DATA.train.batch_size = 32  # Training data batch size
    _C.DATA.train.mask_ratio = 0.0  # Mask out some point features

    _C.DATA.train.return_iterator = False  # Return the data iterator
    _C.DATA.train.return_pts = False  # Also return points

    return _C


class DatasetTest(tf.test.TestCase):

    def test_data(self):
        flags = make_flags()
        octree, label = DatasetFactory(flags.DATA.train)()
        print("octree: ", octree)
        with self.cached_session(use_gpu=True):
            octree_1 = octree.eval()
            print(octree_1, len(octree_1))
        self.assertTrue(octree is not None)


if __name__ == "__main__":
    tf.test.main()
