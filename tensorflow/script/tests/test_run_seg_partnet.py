import numpy as np
import tensorflow as tf

from run_seg_partnet import tf_IoU_per_shape, result_callback, result_callback_maria, get_probabilities


def test_tf_iou_per_shape():
    # following logit and label are for one 3D model that belongs in category C1 which has 3 parts
    logit = tf.Variable(initial_value=np.array([
        # logit for Undefined, C1Part1, C1Part2, C2Part3
        [               0.9,      0.,       0.,      0.],  # logit for point/patch 1
        [               0.,       0.9,      0.,      0.],  # logit for point/patch 2
        [               0.9,      0.,       0.,      0.],  # logit for point/patch 3
        [               0.,       0.,       0.9,     0.],  # logit for point/patch 4
        [               0.,       0.,       0.,      0.9]  # logit for point/patch 5
    ]))
    label = tf.Variable(initial_value=np.array([
        -1,  # label for point/patch 1 MUST BE IGNORED
        1,   # label for point/patch 2 is C1Part1
        0,   # label for point/patch 3 is Undefined
        1,   # label for point/patch 4 is C1Part1
        3    # label for point/patch 5 is C1Part3
    ]))
    class_num = 4  # max(label) + 1
    ignore = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        intersections, unions = sess.run(tf_IoU_per_shape(logit, label, class_num, mask=-1, ignore=ignore))
        assert intersections[ignore] == 0 and unions[ignore] == 0
        assert len(intersections) == class_num
        assert np.array_equal(intersections, np.array([0, 1, 0, 1]))
        assert np.array_equal(unions, np.array([0, 2, 1, 1]))


def test_result_callback():
        # number of points intersected or unioned over all the 3D models for each class
        intersections_and_unions = {
            'intsc_0': 0.0, 'union_0': 0.0,  # iou = 0
            'intsc_1': 20.0, 'union_1': 50.0,  # iou = 2/5
            'intsc_2': 0.0, 'union_2': 30.0,  # iou = 0
            'intsc_3': 0.0, 'union_3': 0.0,  # iou = 0
            'intsc_4': 40.0, 'union_4': 60.0  # iou = 2/3
        }
        # sum of ious over the 4 labels = 16/15
        expected_iou = (16 / 15) / (5-1)
        assert result_callback_maria(intersections_and_unions, 5)['iou'] == expected_iou


def test_get_probabilities():
    logits = tf.Variable(initial_value=np.array([
        # logit for Undefined, C1Part1, C1Part2, C2Part3
        [               1.9,      0.,       0.,      0.],  # logit for point/patch 1
        [               0.,       2.9,      0.,      0.],  # logit for point/patch 2
        [               0.9,      0.8,       0.,      0.],  # logit for point/patch 3
        [               0.,       0.9,       3.9,     0.],  # logit for point/patch 4
        [               0.,       0.,       0.5,      0.9]  # logit for point/patch 5
    ]))
    probs = get_probabilities(logits)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        probabilities = sess.run(probs)
        for point_probs in probabilities:
            assert 9 <= int(np.sum(point_probs) * 10) <= 10
