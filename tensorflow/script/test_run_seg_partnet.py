import numpy as np
import tensorflow as tf

from run_seg_partnet import tf_IoU_per_shape

"""
Note:
If category C1 has 4 parts, then we generate 5 categories: Undefined, C1Part1, C1Part2, C2Part3
The labels we get out of points_property(property_name='label') contain numbers from 0 up to 4 i.e. all the 5 categories. 
However, these labels can be less than 10K, and then the coordinates we get out of points_property(property_name='xyz') are also less than 10K.

In case where a label is -1 (WHEN IS THIS? I DIDNT SEE THIS IN OUR TEST DATA) we ignore it in the evaluation.
"""


def test_tf_io_u_per_shape():
    # following logit and label are for one shape that belongs in category C1 which has 3 parts
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
    class_num = 4
    mask = -1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # iou_debug = sess.run(tf_IoU_per_shape(logit, label, class_num, mask, debug=True))
        # for key, value in iou_debug.items():
        #     print(key, value)
        intersections, unions = sess.run(tf_IoU_per_shape(logit, label, class_num, mask))
        assert np.array_equal(intersections, np.array([1, 1, 0, 1]))
        assert np.array_equal(unions, np.array([1, 2, 1, 1]))
