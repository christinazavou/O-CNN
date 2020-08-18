import sys

sys.path.append("../..")
from src.data_parsing import *
from libs import *


class DatasetDebug:
    classification_channels = {'split': 0, 'label': 0, 'feature': 3, 'index': 1, 'xyz': 1}
    shape_completion_channels = {'split': 1, 'label': 1, 'feature': 3, 'index': 1, 'xyz': 1}

    @staticmethod
    def check_properties():
        # filename = 'resources/ModelNetOnly4Samples3/m40_5_2_12_test_octree.tfrecords'
        # octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
        # octree, label, filename = octrees(filename, 32, shuffle_size=False, return_iterator=False, take=10)
        # filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
        # octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
        # octree, label, filename = octrees(filename, 32, shuffle_size=False, return_iterator=False, take=10)

        # filename = '/media/christina/Data/ANFASS_data/O-CNN/ModelNet40/m40_test_points.tfrecords'
        # depth = 5
        filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
        depth = 6
        octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                      TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                      jitter=0.0, angle=[180, 180, 180],
                                                      bounding_sphere=bounding_sphere),
                                      Points2Octree(depth=depth))
        octree, label = octrees(filename, batch_size=1, shuffle_size=False, return_iterator=False, take=10)
        octree10, label10 = octrees(filename, batch_size=5, shuffle_size=False, return_iterator=False, take=10)
        channels_dict = DatasetDebug.classification_channels

        with tf.Session() as sess:
            # gives error if channel!=0
            for d in range(-10, 10):
                property_name = 'split'
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.float32))
                print("depth {} {} {}".format(d, property_name, result.shape))
                property_name = 'label'
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.float32))
                print("depth {} {} {}".format(d, property_name, result.shape))

            # gives error if channel!=3
            property_name = 'feature'
            for d in range(1, depth + 1):
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.float32))
                print("depth {} {} {}".format(d, property_name, result.shape))

            result = sess.run(octree_property(octree, property_name=property_name, depth=-6,
                                              channel=channels_dict[property_name], dtype=tf.float32))
            print("depth {} {} {}".format(d, property_name, result.shape))

            # gives error if channel!=1
            property_name = 'index'
            for d in range(-1, 8):
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.int32))
                print("depth {} {} {}".format(d, property_name, result.shape))

            # gives error if channel!=1
            property_name = 'xyz'  # this is the shuffle key
            for d in range(-1, 8):
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.uint32))
                print("depth {} {} {}".format(d, property_name, result.shape))

            result = sess.run(octree_property(octree10, property_name=property_name, depth=0,
                                              channel=channels_dict[property_name], dtype=tf.uint32))
            print("octree10: depth {} {} {}".format(0, property_name, result.shape))


DatasetDebug.check_properties()
