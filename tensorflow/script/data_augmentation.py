import sys
import tensorflow as tf
import math

sys.path.append("..")
from libs import *

PI = math.pi


class Points2Octree:
    def __init__(self, depth, full_depth=2, node_dis=False, node_feat=False,
                 split_label=False, adaptive=False, adp_depth=4, th_normal=0.1,
                 save_pts=False, **kwargs):
        self.depth = depth
        self.full_depth = full_depth
        self.node_dis = node_dis  # NOTE: If you want feature channel 4 this needs to be True, otherwise False
        self.node_feat = node_feat
        self.split_label = split_label
        self.adaptive = adaptive
        self.adp_depth = adp_depth
        self.th_normal = th_normal
        self.save_pts = save_pts

    def __call__(self, points, *args):
        octree = points2octree(in_points=points, depth=self.depth, full_depth=self.full_depth,
                               node_dis=self.node_dis, node_feature=self.node_feat,
                               split_label=self.split_label, adaptive=self.adaptive,
                               adp_depth=self.adp_depth, th_normal=self.th_normal,
                               save_pts=self.save_pts)
        return octree


def rotate_point_cloud_and_normals_tf(points, normals, angle=0.0):
    """
        Rotate the point cloud along up direction with certain angle.
        Input:
          N x M array, original point clouds with normal
          scalar, angle of rotation in radians
        Return:
          N x M array, rotated point clouds with normal
    """

    cosval = tf.cos(angle)
    sinval = tf.sin(angle)

    rotation_matrix = tf.convert_to_tensor([[cosval, 0., sinval],
                                            [0., 1., 0.],
                                            [-sinval, 0., cosval]], dtype=tf.float32)

    # !!!!apply rotation on points and normals!!!!
    rot_points = tf.matmul(points, rotation_matrix)
    rot_normals = tf.matmul(normals, rotation_matrix)
    return rot_points, rot_normals


def translate_point_cloud_tf(points, sigma, clip, mu=0.0):
    """
        Globally translate point cloud on x-,y- and z-axis based on a Gaussian distribution,
        with standard deviation sigma and median mu. The noise  values are clamped in the
        interval[-clip,clip].
      :param points: N x 3, numpy.array(float32)
      :param sigma: scalar, float
      :param clip: scalar, float
      :param mu: scalar, float

      :return:
        translated points
    """

    # find min and max points, compute bounding box diagonal
    max_val = tf.reduce_max(points, reduction_indices=[0])
    min_val = tf.reduce_min(points, reduction_indices=[0])
    diagonal = tf.norm(max_val - min_val)

    # compute noise for each input point cloud
    shift = tf.clip_by_value(tf.random.normal([3], mean=mu, stddev=sigma * diagonal), clip_value_min=-clip,
                             clip_value_max=clip)
    points += shift
    return points


def return_self(pts, nrms):
    return pts, nrms


def data_augmentation(points, normals, clip, sigma=0.0, angle=0.0):
    """
        Apply data augmentation in input
         :param points: N x 3, numpy.array(float32)
         :param normals: N x 3, numpy.array(float32)
         :param sigma: scalar, float
         :param clip: scalar, float
         :param angle: scalar, float

         :return:
           augmented points and normals
       """

    # copy point cloud/normals to keep original intact
    pts = tf.identity(points)
    nrms = tf.identity(normals)

    pts, nrms = tf.cond(angle > 0, lambda: rotate_point_cloud_and_normals_tf(pts, nrms, angle),
                        lambda: return_self(pts, nrms))  # rotation
    if sigma > 0:  # global translation
        pts = translate_point_cloud_tf(pts, sigma, clip)
    return pts, nrms


def merge_octrees(octrees, *args):
    octree = octree_batch(octrees)
    return (octree,) + args


class DataAugmentor:

    def get_octree(self, args):
        """
        pts: args[0]
        nrms: args[1]
        fts: args[2]
        pts_labels: args[3]
        rotation: args[-1]
        """
        aug_pts, aug_nrms = data_augmentation(args[0], args[1], self.flags.sigma, self.flags.clip, self.theta * args[-1])
        ocnn_pts = points_new(aug_pts, aug_nrms, args[2], args[3])
        return self.points2octree(ocnn_pts), ocnn_pts

    def __init__(self, flags):
        self.flags=flags
        self.theta = 2 * PI / self.flags.rot_num  # rotation angle in radians
        self.points2octree = Points2Octree(**flags)

    def __call__(self, *args):
        batch = tf.map_fn(lambda x: self.get_octree(x), args,
                          dtype=(tf.string, tf.string))

        return merge_octrees(batch[0]), batch[1]