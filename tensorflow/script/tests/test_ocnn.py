import tensorflow as tf
import numpy as np
import pytest
from scipy.special import softmax
from numpy.random import randint

from ocnn import loss_functions_seg


def random_one_hot_encoding(N, C):
    """ Return a random 2D one-hot vector based on N and C"""
    arr = np.random.randint(low=0, high=C - 1, size=N)
    one_hot = np.zeros((arr.size, C))
    one_hot[np.arange(arr.size), arr] = 1
    return one_hot


def softmax_cross_entropy(onehot_labels, logits, reduction='sum'):
    """Softmax Cross Entropy Numpy implementation"""
    assert (onehot_labels.ndim >= 2)
    assert (onehot_labels.shape == logits.shape)

    # Apply softmax on logits
    softmax_logits = softmax(logits, axis=-1)

    # Reshape is needed
    C = onehot_labels.shape[-1]
    P = onehot_labels.view()
    Q = softmax_logits.view()
    P = np.reshape(P, (-1, C))
    Q = np.reshape(Q, (-1, C))

    # Calculate cross entropy
    H = []
    for p, q in zip(P, Q):
        p_cond = p != 0
        p_val = p[p_cond]
        q_val = q[p_cond]
        assert (q_val.size == 1)
        H.append(-p_val * np.log(q_val))

    # Return H based on reduction mode
    H = np.squeeze(np.asarray(H))
    if reduction == None:
        return H
    elif reduction == 'sum':
        return np.sum(H)
    elif reduction == 'mean':
        return np.mean(H)
    else:
        print("Undefined reduction method {mthd:s}".format(mthd=reduction))
        exit(-1)


def test_tf_same_as_np():
    with tf.Session() as sess:
        for test_idx in range(5):
            # Create 2D or 3D arrays/tensors
            if np.random.random() < 0.5:
                shape = (randint(low=5, high=100), randint(low=5, high=100), randint(low=5, high=100))
            else:
                shape = (randint(low=5, high=100), randint(low=5, high=100))

            # Weighted or unweighted loss
            weights = None
            if np.random.random() < 0.5:
                weights = np.random.rand(shape[-1]) * 10

            # Declare gt labels in one-hot vector encoding and dummy network output (logits)
            if len(shape) == 3:
                onehot_labels = np.zeros(shape=shape, dtype=np.float32)
                for batch_idx in range(shape[0]):
                    labels = random_one_hot_encoding(N=shape[-2], C=shape[-1])
                    if weights is not None:
                        labels = np.multiply(labels, weights)
                    onehot_labels[batch_idx] = labels
            else:
                onehot_labels = random_one_hot_encoding(N=shape[-2], C=shape[-1])
                if weights is not None:
                    onehot_labels = np.multiply(onehot_labels, weights)
            logits = np.random.rand(*shape)
            offset = np.random.randint(low=1, high=10)
            logits = logits * offset - (offset / 2)

            # tf loss
            onehot_labels_pl = tf.placeholder(shape=shape, dtype=tf.float32, name='onehot_labels')
            logits_pl = tf.placeholder(shape=shape, dtype=tf.float32, name='logits')
            loss_op = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels_pl, logits=logits_pl,
                                                      reduction='weighted_mean')
            loss_op1 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels_pl, logits=logits_pl,
                                                       reduction='weighted_sum_by_nonzero_weights')
            tf_loss = sess.run(loss_op, feed_dict={onehot_labels_pl: onehot_labels, logits_pl: logits})
            tf_loss1 = sess.run(loss_op1, feed_dict={onehot_labels_pl: onehot_labels, logits_pl: logits})

            # np loss
            np_loss = softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, reduction='mean')

            assert (np.isclose(tf_loss, np_loss))
            assert (np.isclose(tf_loss1, np_loss))


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
    dc1 = loss_functions_seg(logit=logit, label_gt=label_gt, num_class=num_class, var_name=var_name,
                             weight_decay=0.005, weights=tf.Variable([1., 2, 3, 4]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dc_n = sess.run(dc)
        assert dc_n['accu'] == 0.5
        print(dc_n['loss'])
        dc1_n = sess.run(dc1)
        print(dc_n)
        print(dc1_n)
        # assert np.sum(dc_n['confusion_matrix']) == 2


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
        assert l_weights_n.shape == (6,)


def test_weighted_softmax_loss():
    label_gt = tf.Variable(initial_value=np.array([1, 2]))
    weights = tf.Variable([1., 2, 3, 4])
    unit_weights = tf.Variable([1., 1, 1, 1])
    logit = tf.Variable(
        initial_value=np.array([
            [1, 1.5, 0.2, 0.6],
            [1, 0.5, 0.2, 0.6]
        ]),
        name="ocnn/logits", trainable=True)

    labels = tf.cast(label_gt, tf.int32)
    onehot = tf.one_hot(labels, depth=4)
    l_weights = tf.gather(params=weights, indices=labels)
    l_unit_weights = tf.gather(params=unit_weights, indices=labels)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot, logits=logit, label_smoothing=0.0, weights=l_weights)
    unit_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot, logits=logit, label_smoothing=0.0, weights=l_unit_weights)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_n, unit_loss_n, onehot_n, logit_n, weights_n, unit_weights_n = sess.run(
            [loss, unit_loss, onehot, logit, weights, unit_weights])
        cus_loss = softmax_cross_entropy(onehot_n * weights_n, logit_n, reduction='mean')
        cus_unit_loss = softmax_cross_entropy(onehot_n * unit_weights_n, logit_n, reduction='mean')
        assert np.isclose(cus_unit_loss, unit_loss_n)
        assert np.isclose(cus_loss, loss_n)