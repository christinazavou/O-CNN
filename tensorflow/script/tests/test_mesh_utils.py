from evaluation.mesh_utils import compute_face_centers, nearest_neighbour_of_face_centers
import numpy as np


def test_compute_face_centers():
    vertices = np.array([[1, 1, 1], [-1, -1, -1], [2, 2, 2]])
    faces = np.array([[0, 1, 2], [2, 1, 0], [1, 0, 2]])
    unsampled = [0, 1, 2]
    centers = compute_face_centers(faces=faces, unsampled=unsampled, vertices=vertices)

    assert np.array_equal(centers[0], centers[1]) and np.array_equal(centers[0], centers[2]) and centers.shape == (3, 3)


def test_nearest_neighbour_of_face_centers():
    points = np.array([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]])
    face_centers = np.array([[0.67, 0.67, 0.67]])
    face_feat_from_tr_avg_pool = np.zeros(
        (1, 3))
    face_point_index = np.zeros((1, 1))
    point_feat = np.array([[3, 3, 3], [1, 1, 1], [2, 2, 2]])
    face_indices = [0]
    nearest_neighbour_of_face_centers(face_centers, face_feat_from_tr_avg_pool,
                                      face_point_index, point_feat, points, face_indices)

    assert np.array_equal(face_feat_from_tr_avg_pool, np.array([[2, 2, 2]])) and face_point_index[0] == 2


def test_np_copy():
    x = np.array(range(5))
    y = np.copy(x)
    y += 1

    assert not np.array_equal(x, y) and np.array_equal(x,np.array(range(5)))
