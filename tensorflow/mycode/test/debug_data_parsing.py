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
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth))
    return octrees, filename, depth, task


def config_points_2():
    filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
    depth = 6
    task = 'ae'
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, split_label=True))
    return octrees, filename, depth, task


def config_points_3():
    filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
    depth = 6
    task = 'ae_points_node_dis'
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, split_label=True, node_dis=True))
    return octrees, filename, depth, task


def config_points_4():
    filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
    depth = 6
    task = 'ae_points_adaptive'
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, split_label=True, adaptive=True))
    return octrees, filename, depth, task


def config_points_5():
    filename = '/media/christina/Data/ANFASS_data/O-CNN/shapenet_segmentation/datasets/02691156_test.tfrecords'
    depth = 6
    task = 'seg_points_node_dis'
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                  jitter=0.0, angle=[180, 180, 180],
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, split_label=True, node_dis=True))
    return octrees, filename, depth, task


def config_points_6():
    filename = '/media/christina/Data/ANFASS_data/partnet_data/dataset/Bottle_train_level3.tfrecords'
    depth = 6
    task = 'seg_points_partnet'
    octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                  TransformPoints(distort=True, depth=depth, offset=0, axis='y', scale=0.25,
                                                  jitter=0.125, angle=[5, 5, 5], uniform=True,
                                                  bounding_sphere=bounding_sphere),
                                  Points2Octree(depth=depth, node_dis=True))
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
        },
        'seg_points_partnet': {
            'split': 0, 'label': 1, 'feature': 4, 'index': 1, 'xyz': 1
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
            DatasetDebug.check_d(octree, property_name, d, max_depth, task, session)
        # if property_name == "label":
        #     labels_max_depth = DatasetDebug.check_d(octree, property_name, max_depth, max_depth, task, session)
        #     labels_minus_depth = DatasetDebug.check_d(octree, property_name, -1, max_depth, task, session)

    @staticmethod
    def check_d(octree, property_name, d, max_depth, task, session):
        result = session.run(octree_property(octree, property_name=property_name, depth=d,
                                             dtype=DatasetDebug.dtypes[property_name],
                                             channel=DatasetDebug.channels[task][property_name]))
        print("depth {} {} {}".format(d, property_name, result.shape))
        assert result.shape[0] == DatasetDebug.channels[task][property_name]
        assert d == -1 or result.shape[1] <= 8 ** d
        # if property_name == "label" and 0 < d < max_depth and task == 'seg_points_partnet':
        #     assert set(result.reshape((-1))) <= {-1, 1}  # if d=max_depth then we have label in (-1, #categories)
        # return result

    @staticmethod
    def check_config(octree, octree5, depth, task, sess=None):
        if sess is None:
            sess = tf.Session()
        DatasetDebug.check(octree, 'split', depth, task, sess)
        DatasetDebug.check(octree, 'label', depth, task, sess)

        # "feature" must be the input signal..i.e. in last depth is the nx,ny,nz and then in each preceding
        # depth is the average of its children nodes
        DatasetDebug.check(octree, 'feature', depth, task, sess)
        DatasetDebug.check(octree, 'index', depth, task, sess)

        # "xyz" is the shuffle key
        DatasetDebug.check(octree, "xyz", depth, task, sess)

        try:
            DatasetDebug.check_d(octree5, "xyz", 0, depth, task, sess)
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

    octrees, filename, depth, task = config_points_6()
    with tf.Session() as sess:
        octree, _, points = sess.run(octrees(filename, batch_size=1, shuffle_size=0, return_iterator=True, take=10,
                                             return_pts=True).get_next())
        octree5, _, _ = sess.run(octrees(filename, batch_size=5, shuffle_size=0, return_iterator=True, take=10,
                                         return_pts=True).get_next())
        DatasetDebug.check_config(octree, octree5, depth, task, sess)
        points_xyz = points_property(points, property_name='xyz', channel=4)
        points_label = points_property(points, property_name='label', channel=1)
        points_normal = points_property(points, property_name='normal', channel=3)
        points_feature = points_property(points, property_name='feature', channel=0)
        octree_xyz = octree_property(octree, property_name="xyz", depth=6, dtype=tf.uint32, channel=1)
        octree_label = octree_property(octree, property_name="label", depth=6, dtype=tf.float32, channel=1)
        sess.run([points_xyz, points_label, octree_xyz, octree_label])


check_properties()
