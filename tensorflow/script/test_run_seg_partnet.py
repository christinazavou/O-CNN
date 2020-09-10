import numpy as np
import tensorflow as tf

from run_seg_partnet import tf_IoU_per_shape, PartNetSolver, result_callback, result_callback_maria

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
    mask = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # iou_debug = sess.run(tf_IoU_per_shape(logit, label, class_num, mask, debug=True))
        # for key, value in iou_debug.items():
        #     print(key, value)
        intersections, unions = sess.run(tf_IoU_per_shape(logit, label, class_num, mask))
        assert np.array_equal(intersections, np.array([0, 1, 0, 1]))
        assert np.array_equal(unions, np.array([0, 2, 1, 1]))

        result_dict = {'intsc_%d'%i: intersections[i] for i in range(class_num)}
        result_dict.update({'union_%d'%i: unions[i] for i in range(4)})
        # result_dict = result_callback(result_dict, class_num)
        # assert int(result_dict["iou"]*1000) == 499
        # result_dict = result_callback_maria(result_dict, class_num)
        # assert int(result_dict["iou"]*1000) == 499
        # print(result_dict['iou'])

        result_dict = {'batch': 0.0, 'loss': 2.1302, 'accu': 0.3495, 'regularizer': 2.9788, 'total_loss': 5.109, 'iou': 0.2645, 'intsc_0': 0.0, 'union_0': 0.0, 'intsc_1': 26593.0, 'union_1': 56485.0, 'intsc_2': 0.0, 'union_2': 4996.0, 'intsc_3': 0.0, 'union_3': 0.0, 'intsc_4': 4584.0, 'union_4': 13736.0, 'intsc_5': 94.0, 'union_5': 25000.0, 'intsc_6': 0.0, 'union_6': 655.0, 'intsc_7': 0.0, 'union_7': 16183.0, 'intsc_8': 0.0, 'union_8': 0.0, 'intsc_9': 0.0, 'union_9': 15090.0, 'intsc_10': 0.0, 'union_10': 0.0, 'intsc_11': 0.0, 'union_11': 1443.0, 'intsc_12': 0.0, 'union_12': 0.0, 'intsc_13': 0.0, 'union_13': 0.0, 'intsc_14': 3650.0, 'union_14': 14611.0, 'intsc_15': 0.0, 'union_15': 1832.0, 'intsc_16': 0.0, 'union_16': 0.0, 'intsc_17': 0.0, 'union_17': 137.0, 'intsc_18': 0.0, 'union_18': 0.0, 'intsc_19': 0.0, 'union_19': 5943.0, 'intsc_20': 0.0, 'union_20': 0.0, 'intsc_21': 0.0, 'union_21': 0.0, 'intsc_22': 0.0, 'union_22': 0.0, 'intsc_23': 0.0, 'union_23': 0.0, 'intsc_24': 0.0, 'union_24': 7649.0, 'intsc_25': 0.0, 'union_25': 1137.0, 'intsc_26': 0.0, 'union_26': 0.0, 'intsc_27': 0.0, 'union_27': 0.0, 'intsc_28': 0.0, 'union_28': 0.0, 'intsc_29': 0.0, 'union_29': 0.0, 'intsc_30': 0.0, 'union_30': 0.0, 'intsc_31': 0.0, 'union_31': 0.0, 'intsc_32': 0.0, 'union_32': 0.0, 'intsc_33': 0.0, 'union_33': 0.0}

        assert int(result_callback_maria(result_dict, 34)['iou']*10000) == 2645
