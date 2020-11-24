import sys

from config import override_some_flags

sys.path.append("..")
from libs import *
from dataset import *

categories = ['Buildings']
labels = [34],
take = [200]


def config_points_buildings_with_colour():
    # data generated with python only
    filename = '/media/christina/Data/ANNFASS_data/O-CNN/data/test_w_colour_no_rot.tfrecords'
    depth = 7
    task = 'seg_points_buildings_w_colour'
    octrees = PointDataset(ParseExample(x_alias='data', y_alias='label'),
                           NormalizePoints(),
                           CustomTransformPoints(),
                           Points2Octree(depth=depth, node_dis=True, ))
    return octrees, filename, depth, task


def config_points_buildings_no_colour():
    # data generated with python only
    filename = '/media/christina/Data/ANNFASS_data/O-CNN/data/test_no_colour_no_rot.tfrecords'
    depth = 7
    task = 'seg_points_buildings_n_colour'
    octrees = PointDataset(ParseExample(x_alias='data', y_alias='label'),
                           NormalizePoints(),
                           CustomTransformPoints(),
                           Points2Octree(depth=depth, node_dis=False, ))
    return octrees, filename, depth, task


def config_points_buildings_with_colour_filenames():
    # data generated with python only
    filename = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset_points_chunk8/train.shuffle.all.tfrecords'
    depth = 7
    task = 'seg_points_buildings_w_colour'
    octrees = PointDatasetDebug(ParseExampleDebug())
    return octrees, filename, depth, task


class DatasetDebug:
    channels = {
        'seg_points_partnet': {
            'split': 0, 'label': 1, 'feature': 4, 'index': 1, 'xyz': 1
        },
        'seg_points_buildings': {
            'split': 0, 'label': 1, 'feature': 4, 'index': 1, 'xyz': 1
        },
        'seg_points_buildings_w_colour': {
            'split': 0, 'label': 1, 'feature': 8, 'index': 1, 'xyz': 1
            # FIXME: TODO: ask why now features are 8 and not 11
            # nz,ny,nz,dis,r,g,b,a
        },
        'seg_points_buildings_n_colour':{
            'split': 0, 'label': 1, 'feature': 3, 'index': 1, 'xyz': 1
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

    @staticmethod
    def check_d(octree, property_name, d, max_depth, task, session):
        result = session.run(octree_property(octree, property_name=property_name, depth=d,
                                             dtype=DatasetDebug.dtypes[property_name],
                                             channel=DatasetDebug.channels[task][property_name]))
        print("depth {} {} {}".format(d, property_name, result.shape))
        assert result.shape[0] == DatasetDebug.channels[task][property_name]
        assert d == -1 or result.shape[1] <= 8 ** d

    @staticmethod
    def check_config(octree, octree5, depth, task, sess=None):
        if sess is None:
            sess = tf.Session()
        DatasetDebug.check(octree, 'split', depth, task, sess)
        DatasetDebug.check(octree, 'label', depth, task, sess)

        DatasetDebug.check(octree, 'feature', depth, task, sess)
        DatasetDebug.check(octree, 'index', depth, task, sess)

        DatasetDebug.check(octree, "xyz", depth, task, sess)

        try:
            DatasetDebug.check_d(octree5, "xyz", 0, depth, task, sess)
        except:
            pass  # more than one octrees merged thus more output rows in the result


def check_filenames():
    with tf.Session() as sess:
        octrees, filename, depth, task = config_points_buildings_with_colour_filenames()
        count = 0
        filepaths = sess.run(octrees(filename, batch_size=1600))
        for filepath in filepaths:
            filepath = filepath.decode('utf8')
            count += 1
            if count == 900:
                assert "RESIDENTIALhouse_mesh2627_w_label" in filepath, filepath

        x = octrees(filename, batch_size=300)
        filepaths = sess.run(x)
        filepaths = sess.run(x)
        filepath = filepaths[-1].decode('utf8')
        assert 'RESIDENTIALchurch_mesh1845_w_label' in filepath, filepath


def check_dataset_properties_with_colour():

    filename = '/media/christina/Data/ANNFASS_data/O-CNN/data/test_w_colour_no_rot.tfrecords'
    flags = override_some_flags(filename)

    dataset_iter = DatasetFactory(flags.DATA.test)(return_iter=True)

    with tf.Session() as sess:
        for i in [0,1,2]:
            octree, labels, points = dataset_iter.get_next()

            pts_init_n, labels_n, normal_n = sess.run([
                points_property(points, property_name='xyz', channel=4),
                points_property(points, property_name='label', channel=1),
                points_property(points, property_name='normal', channel=3),
            ])

            print("pts_init_n: ", pts_init_n[0:3], "\n")
            print("labels_n: ", labels_n[0:3], "\n")
            print("normal_n: ", normal_n[0:3], "\n")

            of7 = sess.run(octree_property(octree, property_name='feature', dtype=tf.float32, depth=7, channel=8))
            print("of7: ", of7, "\n")
            l = sess.run(octree_property(octree, property_name='label', dtype=tf.float32, depth=7, channel=1))
            print("l: ", l, "\n")
            xyz = sess.run(octree_property(octree, property_name='xyz', dtype=tf.uint32, depth=7, channel=1))
            print("xyz: ", xyz, "\n")


def check_dataset_properties_no_colour():

    filename = '/media/christina/Data/ANNFASS_data/O-CNN/data/test_no_colour_no_rot.tfrecords'
    flags = override_some_flags(filename)

    dataset_iter = DatasetFactory(flags.DATA.test)(return_iter=True)

    with tf.Session() as sess:
        for i in [0,1,2]:
            octree, labels, points = dataset_iter.get_next()

            pts_init_n, labels_n, normal_n = sess.run([
                points_property(points, property_name='xyz', channel=4),
                points_property(points, property_name='label', channel=1),
                points_property(points, property_name='normal', channel=3),
            ])

            print("pts_init_n: ", pts_init_n[0:3], "\n")
            print("labels_n: ", labels_n[0:3], "\n")
            print("normal_n: ", normal_n[0:3], "\n")

            of = sess.run(octree_property(octree, property_name='feature', dtype=tf.float32, depth=7, channel=3))
            print("of: ", of, "\n")
            l = sess.run(octree_property(octree, property_name='label', dtype=tf.float32, depth=7, channel=1))
            print("l: ", l, "\n")
            xyz = sess.run(octree_property(octree, property_name='xyz', dtype=tf.uint32, depth=7, channel=1))
            print("xyz: ", xyz, "\n")


def check_properties():
    octrees, filename, depth, task = config_points_buildings_with_colour()
    # octrees, filename, depth, task = config_points_buildings_no_colour()

    with tf.Session() as sess:

        octree, _, points = sess.run(octrees(filename, batch_size=1, shuffle_size=0, return_iter=True, take=10,
                                             return_pts=True).get_next())
        octree5, _, points5 = sess.run(octrees(filename, batch_size=5, shuffle_size=0, return_iter=True, take=10,
                                               return_pts=True).get_next())
        DatasetDebug.check_config(octree, octree5, depth, task, sess)
        points_xyzd = points_property(points, property_name='xyz', channel=4)
        points_xyz = points_property(points, property_name='xyz', channel=3)
        points_label = points_property(points, property_name='label', channel=1)
        octree_xyz = octree_property(octree, property_name="xyz", depth=depth, dtype=tf.uint32, channel=1)
        octree_label = octree_property(octree, property_name="label", depth=depth, dtype=tf.float32, channel=1)
        pxyzd, pxyz, plabel, oxyz, olabel = sess.run([points_xyzd, points_xyz, points_label, octree_xyz, octree_label])
        print("points_xyzd: ", pxyzd.shape)
        print("points_xyz: ", pxyz.shape)
        print("points_label: ", plabel.shape)
        print("octree_xyz: ", oxyz.shape)
        print("octree_label: ", olabel.shape)

        points_normal = points_property(points, property_name='normal', channel=3)
        pnormal = sess.run(points_normal)
        print("points_normal: ", pnormal.shape)

        octree_feature = octree_property(octree, property_name="feature", depth=depth, dtype=tf.float32,
                                         # channel=DatasetDebug.channels[task]['feature'])
                                         channel=7)
        ofeature = sess.run(octree_feature)
        print("octree_feature: ", ofeature.shape)

        points5_xyzd_tf = points_property(points5, property_name='xyz', channel=4)
        print("points5_xyzd ", sess.run(points5_xyzd_tf).shape)


# check_filenames()
# check_properties()
check_dataset_properties_with_colour()
# check_dataset_properties_no_colour()
