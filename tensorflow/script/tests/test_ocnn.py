import tensorflow as tf
import numpy as np
import pytest
from ocnn import loss_functions_seg


def test_loss_functions_seg():
    logit = tf.Variable(
        initial_value=np.array([
            [0, 0.1, 0.2, 0.3],
            [1, 1.5, 0.2, 0.6],
            [0, 0.1, 0.2, 0.3],
            [1, 0.5, 0.2, 0.6]
        ]),
        name="ocnn/logits", trainable=True)
    label_gt = tf.Variable(initial_value=np.array([-1, 1, 0, 2]))
    num_class = 4
    var_name = 'ocnn'
    dc = loss_functions_seg(logit=logit, label_gt=label_gt, num_class=num_class, var_name=var_name,
                            weight_decay=0.005, weights=tf.Variable([1., 1, 1, 1]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dc_n = sess.run(dc)
        assert dc_n['accu'] == 0.5
        assert np.sum(dc_n['confusion_matrix']) == 2


@pytest.mark.skip(reason="Debugging purposes.")
def test_tf_gather():
    weights = tf.Variable(initial_value=np.array([0, 0.1, 0.2, 0.3]))
    labels = tf.Variable(initial_value=np.array([-1, 1, 0, 2, 3, 1, 2, 3]))
    mask = tf.logical_and(tf.not_equal(labels, -1), tf.not_equal(labels, 0))

    masked_labels = tf.boolean_mask(labels, mask)
    l_weights = tf.gather(params=weights, indices=masked_labels)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l_weights_n = sess.run(l_weights)
        assert l_weights_n.shape == (6, )
