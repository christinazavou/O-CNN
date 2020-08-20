import sys

sys.path.append("../..")
from src.data_parsing import *
from libs import *


def config_octrees_1():
    octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
    filename, depth, task = '/media/christina/Data/ANFASS_data/O-CNN/ModelNet40/m40_5_2_12_test_octree.tfrecords', 5, 'cls'
    return octrees, filename, depth, task


def config_octrees_2():
    octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
    filename, depth, task = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_octrees.tfrecords', 6, 'ae'
    return octrees, filename, depth, task


def config_octrees_3():
    octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
    filename, depth, task = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_aoctrees.tfrecords', 6, 'ae'
    return octrees, filename, depth, task


def config_points_1():
    filename, depth, task = '/media/christina/Data/ANFASS_data/O-CNN/ModelNet40/m40_test_points.tfrecords', 5, 'cls'
    split_label = False
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, split_label=split_label))
    return octrees, filename, depth, task


def config_points_2():
    filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
    depth = 6
    split_label = True
    task = 'ae'
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, split_label=split_label))
    return octrees, filename, depth, task


def config_points_3():
    filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
    depth = 6
    split_label = True
    node_dis = True
    task = 'ae_points_node_dis'
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, split_label=split_label, node_dis=node_dis))
    return octrees, filename, depth, task


def config_points_4():
    filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
    depth = 6
    split_label = True
    task = 'ae_points_adaptive'
    adaptive = True
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, split_label=split_label, adaptive=adaptive))
    return octrees, filename, depth, task


def config_points_5():
    filename = '/media/christina/Data/ANFASS_data/O-CNN/shapenet_segmentation/datasets/02691156_test.tfrecords'
    depth = 6
    split_label = True
    node_dis = True
    task = 'seg_points_node_dis'
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, split_label=split_label, node_dis=node_dis))
    return octrees, filename, depth, task


class DatasetDebug:
    channels = {
        'cls': {
            'split': 0, 'label': 0, 'feature': 3, 'index': 1, 'xyz': 1
        },
        'ae': {
            'split': 1, 'label': 0, 'feature': 3, 'index': 1, 'xyz': 1
        },
        'ae_points_node_dis': {
            'split': 1, 'label': 0, 'feature': 4, 'index': 1, 'xyz': 1
        },
        'ae_points_adaptive': {
            'split': 1, 'label': 0, 'feature': 3, 'index': 1, 'xyz': 1
        },
        'seg_points_node_dis': {
            'split': 1, 'label': 1, 'feature': 4, 'index': 1, 'xyz': 1
        }
    }

    dtypes = {
        'split': tf.float32,
        'label': tf.float32,
        'feature': tf.float32,
        'index': tf.int32,
        'xyz': tf.uint32
    }

    @staticmethod
    def check(octree, property_name, max_depth, task, session):
        for d in range(0, max_depth + 1):
            DatasetDebug.check_d(octree, property_name, d, task, session)

    @staticmethod
    def check_d(octree, property_name, d, task, session):
        result = session.run(octree_property(octree, property_name=property_name, depth=d,
                                             dtype=DatasetDebug.dtypes[property_name],
                                             channel=DatasetDebug.channels[task][property_name]))
        print("depth {} {} {}".format(d, property_name, result.shape))
        assert result.shape[0] == DatasetDebug.channels[task][property_name]
        assert result.shape[1] <= 8 ** d

    @staticmethod
    def check_config(octree, octree5, depth, task):
        with tf.Session() as sess:
            DatasetDebug.check(octree, 'split', depth, task, sess)
            DatasetDebug.check(octree, 'label', depth, task, sess)

            # "feature" must be the input signal..i.e. in last depth is the nx,ny,nz and then in each preceding
            # depth is the average of its children nodes
            DatasetDebug.check(octree, 'feature', depth, task, sess)
            try:
                DatasetDebug.check_d(octree, 'feature', -6, task, sess)
            except:
                pass  # not meaningful depth
            DatasetDebug.check(octree, 'index', depth, task, sess)

            # "xyz" is the shuffle key
            DatasetDebug.check(octree, "xyz", depth, task, sess)

            try:
                DatasetDebug.check_d(octree5, "xyz", 0, task, sess)
            except:
                pass  # more than one octrees merged thus more output rows in the result


def check_properties():
    # octrees, filename, depth, task = config_octrees_1()
    # octree, label, filenames = octrees(filename, batch_size=1, shuffle_size=0, return_iterator=False, take=10)
    # octree5, label5, filenames5 = octrees(filename, batch_size=5, shuffle_size=0, return_iterator=False, take=10)
    # DatasetDebug.check_config(octree, octree5, depth, task)
    #
    # octrees, filename, depth, task = config_octrees_2()
    # octree, label, filenames = octrees(filename, batch_size=1, shuffle_size=0, return_iterator=False, take=10)
    # octree5, label5, filenames5 = octrees(filename, batch_size=5, shuffle_size=0, return_iterator=False, take=10)
    # DatasetDebug.check_config(octree, octree5, depth, task)
    #
    # octrees, filename, depth, task = config_octrees_3()
    # octree, label, filenames = octrees(filename, batch_size=1, shuffle_size=0, return_iterator=False, take=10)
    # octree5, label5, filenames5 = octrees(filename, batch_size=5, shuffle_size=0, return_iterator=False, take=10)
    # DatasetDebug.check_config(octree, octree5, depth, task)

    # octrees, filename, depth, task = config_points_1()
    # octree, label = octrees(filename, batch_size=1, shuffle_size=0, return_iterator=False, take=10)
    # octree5, label5 = octrees(filename, batch_size=5, shuffle_size=0, return_iterator=False, take=10)
    # DatasetDebug.check_config(octree, octree5, depth, task)

    # octrees, filename, depth, task = config_points_2()
    # octree, label = octrees(filename, batch_size=1, shuffle_size=0, return_iterator=False, take=10)
    # octree5, label5 = octrees(filename, batch_size=5, shuffle_size=0, return_iterator=False, take=10)
    # DatasetDebug.check_config(octree, octree5, depth, task)
    #
    # octrees, filename, depth, task = config_points_3()
    # octree, label = octrees(filename, batch_size=1, shuffle_size=0, return_iterator=False, take=10)
    # octree5, label5 = octrees(filename, batch_size=5, shuffle_size=0, return_iterator=False, take=10)
    # DatasetDebug.check_config(octree, octree5, depth, task)
    #
    # octrees, filename, depth, task = config_points_4()
    # octree, label = octrees(filename, batch_size=1, shuffle_size=0, return_iterator=False, take=10)
    # octree5, label5 = octrees(filename, batch_size=5, shuffle_size=0, return_iterator=False, take=10)
    # DatasetDebug.check_config(octree, octree5, depth, task)
    #
    octrees, filename, depth, task = config_points_5()
    octree, label = octrees(filename, batch_size=1, shuffle_size=0, return_iterator=False, take=10)
    octree5, label5 = octrees(filename, batch_size=5, shuffle_size=0, return_iterator=False, take=10)
    DatasetDebug.check_config(octree, octree5, depth, task)


check_properties()
