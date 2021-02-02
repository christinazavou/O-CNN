import sys
import os
import tensorflow as tf
import numpy as np
import json
import math
from tqdm import tqdm

# import psutil

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


def rotate_point_cloud_and_normals(points, angle=0.0):
    """
        Rotate the point cloud along up direction with certain angle.
        Input:
          N x M array, original point clouds with normal
          scalar, angle of rotation in radians
        Return:
          N x M array, rotated point clouds with normal
    """

    tmp_points = points.copy()
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    tmp_points[..., 0:3] = np.dot(points[..., 0:3].reshape((-1, 3)), rotation_matrix)
    tmp_points[..., 3:6] = np.dot(points[..., 3:6].reshape((-1, 3)), rotation_matrix)
    return tmp_points


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
                                            [-sinval, 0., cosval]])

    # !!!!apply rotation on points and normals!!!!
    rot_points = tf.matmul(points, rotation_matrix)
    rot_normals = tf.matmul(normals, rotation_matrix)
    return rot_points, rot_normals


def translate_point_cloud_numpy(points, sigma, clip, mu=0.0):
    """
        Globally translate point cloud on x-,y- and z-axis based on a Gaussian distribution,
        with standard deviation sigma and median mu. The noise  values are clamped in the
        interval[-clip,clip].
      :param points: N x M, numpy.array(float32)
      :param sigma: scalar, float
      :param clip: scalar, float
      :param mu: scalar, float

      :return:
        translated points
    """
    # find min and max points, compute bounding box diagonal
    max_val = np.max(points, axis=1)
    min_val = np.min(points, axis=1)
    diagonal = np.linalg.norm(max_val - min_val)

    # compute noise for each input point cloud
    shift = np.clip(np.random.normal(mu, sigma * diagonal, (3, 1)), -clip, clip)

    points += shift
    return points


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
    return (pts, nrms)


def data_augmentation(points, normals, clip, sigma=0.0, angle=0.0):
    """
        Apply data augmentation in input
         :param points: N x 3, numpy.array(float32)
         :param normals: N x 3, numpy.array(float32)
         :param sigma: scalar, float
         :param clip: scalar, float
         :param mu: scalar, float

         :return:
           augmented points and normals
       """

    # copy point cloud/normals to keep original intact
    pts = tf.identity(points)
    nrms = tf.identity(normals)

    pts, nrms = tf.cond(angle > 0, lambda: rotate_point_cloud_and_normals_tf(pts, nrms, angle),
                        lambda: return_self(pts, nrms))  # rotation
    print(pts, nrms)
    # pts, nrms =
    if sigma > 0:  # global translation
        pts = translate_point_cloud_tf(pts, sigma, clip)
    return pts, nrms


def merge_octrees(octrees, *args):
    octree = octree_batch(octrees)
    return (octree,) + args


def get_octree(self, args):
    idx = args[0]
    rot = args[1]
    pts = self.points.read(idx)
    nrms = self.normals.read(idx)
    # if features are a scalar they are ignored in octree creation script
    fts = self.features.read(idx) if self.has_features else 0.0
    cl_l = self.class_labels.read(idx)
    pt_l = self.point_labels.read(idx)
    name = self.filenames.read(idx)
    pts, nrms = data_augmentation(pts, nrms, self.flags.sigma, self.flags.clip, self.theta * rot)
    ocnn_pts = points_new(pts, nrms, fts, pt_l)
    return self.points2octree(ocnn_pts), cl_l, ocnn_pts, name


class DataLoader:
    def __init__(self, flags):

        def load_points_file(filename):
            if ".ply" in filename:
                try:
                    p = open(filename, "r").readlines()
                except OSError:
                    print("Could not open file: ", filename)
                    sys.exit()

                while not "end_header" in p[0]:
                    p.pop(0)
                p.pop(0)
                point_clouds = np.array([[float(i) for i in j.strip().split()] for j in p])
            else:  # txt
                try:
                    point_clouds = np.loadtxt(filename).astype(np.float32)
                except ValueError:
                    print("Could not load file: ", filename)
                    sys.exit()

            return point_clouds

        def read_files():
            print("\nReading files. This may take a while...\n")
            try:
                with open(self.flags.file_list, "r") as f:
                    cnt = 0
                    for line in tqdm(f):
                        line = line.strip().split()
                        self.filenames = self.filenames.write(cnt, line[0].split(".")[0])
                        self.class_labels = self.class_labels.write(cnt, int(line[1]))
                        pts = load_points_file(os.path.join(self.flags.location, line[0]))
                        self.points = self.points.write(cnt, pts[..., 0:3])
                        self.normals = self.normals.write(cnt, pts[..., 3:6])
                        if pts.shape[-1] > 6:
                            self.features = self.features.write(cnt, pts[..., 6:])
                        self.point_labels = self.point_labels.write(cnt, np.array(list(json.load(open(
                            os.path.join(self.flags.label_location, line[0].split(".")[0] + "_label.json"))).values()),
                                                                                  dtype=np.float32))
                        cnt += 1
                    self.tfrecord_num = cnt
            except OSError:
                print("Could not open data file list. Exiting...")
                sys.exit()

        self.points = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                     tensor_array_name="tfrecord_xyz")
        self.normals = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                      tensor_array_name="tfrecord_normals")
        self.features = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                       tensor_array_name="tfrecord_features")
        self.filenames = tf.TensorArray(dtype=tf.string, size=0, dynamic_size=True, clear_after_read=False,
                                        tensor_array_name="tfrecord_filenames")
        self.point_labels = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False,
                                           tensor_array_name="tfrecord_point_labels")
        self.class_labels = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True, clear_after_read=False,
                                           tensor_array_name="tfrecord_class_labels")

        self.flags = flags
        self.tfrecord_num = 0
        self.theta = 2 * PI / self.flags.rot_num  # rotation angle in radians
        self.points2octree = Points2Octree(**flags)

        read_files()
        if self.features.element_shape==None:
            self.has_features=False
        else:
            self.has_features=True

        # memory consumption
        # py = psutil.Process(os.getpid())
        # memory_usage = py.memory_info()[0] / math.pow(1024, 3)
        # print(memory_usage);

    def getter(self):
        return self

    def __call__(self, return_iter=True, *args, **kwargs):

        idxs = np.tile(np.arange(self.tfrecord_num), self.flags.rot_num).astype(np.int32)
        rots = np.repeat(np.arange(self.flags.rot_num), self.tfrecord_num).astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((idxs, rots)).take(-1)
        if self.flags.shuffle > 0: dataset = dataset.shuffle(self.tfrecord_num * self.flags.rot_num)
        itr = dataset.repeat().batch(self.flags.batch_size).prefetch(
            self.flags.batch_size * 2).make_one_shot_iterator()

        return itr if return_iter else itr.get_next()
