import tensorflow as tf
import numpy as np


def point_cloud_diagonal_numpy(points):
    """
        Compute point cloud bounding box diagonal
      :param points: K x N x M, numpy.array(float32)

      :return:
        diagonal
    """
    # find min and max points, compute bounding box diagonal

    max_val = np.max(points, axis=1)
    min_val = np.min(points, axis=1)
    diagonal = np.linalg.norm(max_val - min_val)

    return diagonal


def point_cloud_diagonal(points):
    """
        Compute point cloud bounding box diagonal
      :param points: N x 3, numpy.array(float32)

      :return:
        diagonal
    """
    # find min and max points, compute bounding box diagonal
    max_val = tf.reduce_max(points, reduction_indices=[0])
    min_val = tf.reduce_min(points, reduction_indices=[0])
    diagonal = tf.norm(max_val - min_val)

    return diagonal


def test_translate_point_cloud():
    filename = "/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/nocolor/RESIDENTIALhouse_mesh2331.ply"
    p = open(filename, "r").readlines()
    while not "end_header" in p[0]:
        p.pop(0)
    p.pop(0)

    point_cloud = np.array([[float(i) for i in j.strip().split()] for j in p])
    diagonal_np = point_cloud_diagonal_numpy([point_cloud[..., 0:3]])
    diagonal_tf = point_cloud_diagonal(point_cloud[..., 0:3])

    with tf.Session() as sess:
        d_tf = sess.run(diagonal_tf)
    assert np.isclose(diagonal_np, d_tf)
