import sys

sys.path.append("../..")
from src.data_parsing import *
from libs import *


class DatasetDebug:

    @staticmethod
    def check_points_to_ply():
        build_path = "/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/octree/build"
        f = os.getcwd() + "/resources/ocnn_completion_only2samples2/points2ply.txt"
        o = os.getcwd() + "/output/ocnn_completion/points2ply"
        os.makedirs(o)
        os.system("cd {} && ./points2ply --filenames {} --output_path {}".format(build_path, f, o))

    @staticmethod
    def check_properties():
        # filename = 'resources/ModelNetOnly4Samples3/m40_5_2_12_test_octree.tfrecords'
        # octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
        # octree, label, filename = octrees(filename, 32, shuffle_size=False, return_iterator=False, take=10)

        # filename = 'resources/ModelNetOnly4Samples3/m40_train_points.tfrecords'
        # octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
        #                               TransformPoints(distort=False, depth=5, offset=0.55, axis='z', scale=0.0,
        #                                               jitter=0.0, angle=[180, 180, 180],
        #                                               bounding_sphere=bounding_sphere),
        #                               Points2Octree(depth=5))
        # octree, label = octrees(filename, 32, shuffle_size=False, return_iterator=False, take=10)

        filename = 'resources/ocnn_completion_only2samples2/completion_test_octrees.tfrecords'
        octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
        octree, label, filename = octrees(filename, 32, shuffle_size=False, return_iterator=False, take=10)

        with tf.Session() as sess:
            # it only lets me use channel=0
            for d in range(-10, 10):
                res = sess.run(octree_property(octree, property_name="split", dtype=tf.float32, depth=d, channel=0))
                print("d ", d, " split ", res.shape)
                res = sess.run(octree_property(octree, property_name="label", depth=d, channel=0, dtype=tf.float32))
                print("d ", d, " label ", res.shape)

            # it only lets me use channel=3
            for d in range(1, 6):
                res = sess.run(octree_property(octree, property_name="feature", depth=d, channel=3, dtype=tf.float32))
                print("d ", d, " feature ", res.shape)

            res = sess.run(octree_property(octree, property_name="feature", depth=-6, channel=3, dtype=tf.float32))
            print("d ", -6, " feature ", res.shape)

            # it only lets me use channel=1
            for d in range(-1, 8):
                res = sess.run(octree_property(octree, property_name="index", depth=d, channel=1, dtype=tf.int32))
                print("d ", d, " index ", res.shape)

            # it only lets me use channel=1
            for d in range(-2, 8):
                res = sess.run(octree_property(octree, property_name="xyz", depth=4, channel=1, dtype=tf.uint32))
                print("d ", d, " xyz ", res.shape)


# DatasetDebug.check_points_to_ply()
DatasetDebug.check_properties()
