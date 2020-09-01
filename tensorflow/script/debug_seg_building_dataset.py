import sys

sys.path.append("..")
from libs import *
from dataset import *

categories = [
    'Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher',
    'Display', 'Door', 'Earphone', 'Faucet', 'Hat', 'Keyboard',
    'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator',
    'Scissors', 'StorageFurniture', 'Table', 'TrashCan', 'Vase'
]
labels = [
    -1, 15, 9, -1,
    39, 11, 7, 4,
    5, 10, 12, -1,
    -1, 10, 41, -1,
    6, -1, 7, -1,
    24, 51, 11, 6
]
take = [
    -1, 37, 84, -1,
    1217, 98, 51, 191,
    51, 53, 132, -1,
    -1, 77, 419, -1,
    39, -1, 31, -1,
    451, 1668, 63, 233
]
categories = ['Buildings']
labels = [34],
take = [200]


def config_points_partnet(category="Bottle"):
    filename = '/media/christina/Data/ANFASS_data/partnet_data/dataset/{}_train_level3.tfrecords' \
        .format(category)
    depth = 6
    task = 'seg_points_partnet'

    octrees = PointDataset(ParseExample(x_alias='data', y_alias='label'),
                           NormalizePoints(),
                           TransformPoints(distort=True, depth=depth, offset=0, axis='y',
                                           scale=0.25, jitter=0.125, angle=[5, 5, 5],
                                           uniform=True,
                                           bounding_sphere=bounding_sphere),
                           Points2Octree(depth=depth, node_dis=True))

    return octrees, filename, depth, task


def config_points_buildings():
    # filename = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset_points/test_200points100000.tfrecords'
    filename = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset_points/test_points.tfrecords'
    depth = 7
    task = 'seg_points_buildings'
    octrees = PointDataset(ParseExample(x_alias='data', y_alias='label'),
                           NormalizePoints(),
                           TransformPoints(distort=True, depth=depth, offset=0, axis='y',
                                           scale=0.25, jitter=0.125, angle=[5, 5, 5],
                                           uniform=True,
                                           bounding_sphere=bounding_sphere),
                           Points2Octree(depth=depth, node_dis=True, ))
    return octrees, filename, depth, task


def config_points_buildings():
    filename = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset_points/test_points.tfrecords'
    depth = 7
    task = 'seg_points_buildings'
    octrees = PointDataset(ParseExample(x_alias='data', y_alias='label'),
                           NormalizePoints(),
                           TransformPoints(distort=True, depth=depth, offset=0, axis='y',
                                           scale=0.25, jitter=0.125, angle=[5, 5, 5],
                                           uniform=True,
                                           bounding_sphere=bounding_sphere),
                           Points2Octree(depth=depth, node_dis=True, ))
    return octrees, filename, depth, task


class DatasetDebug:
    channels = {
        'seg_points_partnet': {
            'split': 0, 'label': 1, 'feature': 4, 'index': 1, 'xyz': 1
        },
        'seg_points_buildings': {
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
    # octrees, filename, depth, task = config_points_partnet()
    octrees, filename, depth, task = config_points_buildings()
    with tf.Session() as sess:
        octree, _, points = sess.run(octrees(filename, batch_size=1, shuffle_size=0,
                                             return_iter=True, take=10,
                                             return_pts=True).get_next())
        octree5, _, _ = sess.run(octrees(filename, batch_size=5, shuffle_size=0,
                                         return_iter=True, take=10,
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

        # points_feature = points_property(points, property_name='feature', channel=4)
        # pfeature = sess.run(points_feature)
        # print("points_feature: ", pfeature.shape)

        octree_feature = octree_property(octree, property_name="feature", depth=depth, dtype=tf.float32, channel=4)
        ofeature = sess.run(octree_feature)
        print("octree_feature: ", ofeature.shape)

        # x,y,z,d,nx,ny,nz,r,g,b,a,l

        # ----------------------------------------------------------------------------------

        # for cat_idx in range(len(categories)):
        #     minl = 100
        #     maxl = -1
        #     if labels[cat_idx] == -1:
        #         continue
        #
        #     # octrees, filename, depth, task = config_points_partnet(categories[cat_idx])
        #     octrees, filename, depth, task = config_points_buildings()
        #     _, _, points = octrees(filename, batch_size=1, shuffle_size=0,
        #                            return_iterator=False, take=take[cat_idx], return_pts=True)
        #     points_label = points_property(points, property_name='label', channel=1)
        #     points_xyzd = points_property(points, property_name='xyz', channel=4)
        #
        #     # points_label = points_property(octrees(filename, batch_size=1, shuffle_size=0,
        #     #                                        return_iterator=False, take=take[cat_idx],
        #     #                                        return_pts=True)[2], property_name='label',
        #     #                                channel=1)
        #     for i in range(take[cat_idx]):
        #         pl, xyz = sess.run([points_label, points_xyzd])
        #         minl = min(minl, pl.min())
        #         maxl = max(maxl, pl.max())
        #
        #         if pl.shape[0] != 10000:
        #             assert pl.shape[0] < 10000
        #             assert pl.shape[0] == xyz.shape[0]
        #
        #     print(categories[cat_idx], minl, maxl)


check_properties()
