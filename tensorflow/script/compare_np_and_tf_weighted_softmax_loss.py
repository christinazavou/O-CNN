import tensorflow as tf
from scipy.special import softmax
import numpy as np


# tf.set_random_seed(1)
# np.random.seed(1)

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


def random_one_hot_encoding(N, C):
    """ Return a random 2D one-hot vector based on N and C"""
    arr = np.random.randint(low=0, high=C - 1, size=N)
    one_hot = np.zeros((arr.size, C))
    one_hot[np.arange(arr.size), arr] = 1

    return one_hot


def make_unit_tests(n_tests, sess):
    for test_idx in range(n_tests):
        # Create 2D or 3D arrays/tensors
        if np.random.random() < 0.5:
            shape = (
            np.random.randint(low=5, high=100), np.random.randint(low=5, high=100), np.random.randint(low=5, high=100))
        else:
            shape = (np.random.randint(low=5, high=100), np.random.randint(low=5, high=100))

        # Use weights
        if np.random.random() < 0.5:
            weight_flag = True
            weights = np.random.rand(shape[-1]) * 10
        else:
            weight_flag = False

        # Declare gt labels in one-hot vector encoding and dummy network output (logits)
        if len(shape) == 3:
            onehot_labels = np.zeros(shape=shape, dtype=np.float32)
            for batch_idx in range(shape[0]):
                labels = random_one_hot_encoding(N=shape[-2], C=shape[-1])
                if weight_flag:
                    labels = np.multiply(labels, weights)
                onehot_labels[batch_idx] = labels
        else:
            onehot_labels = random_one_hot_encoding(N=shape[-2], C=shape[-1])
            if weight_flag:
                onehot_labels = np.multiply(onehot_labels, weights)
        logits = np.random.rand(*shape)
        offset = np.random.randint(low=1, high=10)
        logits = logits * offset - (offset / 2)

        # Declare tf.placeholders
        onehot_labels_pl = tf.placeholder(shape=shape, dtype=tf.float32, name='onehot_labels')
        logits_pl = tf.placeholder(shape=shape, dtype=tf.float32, name='logits')

        # Define loss in the computational graph
        loss_op = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels_pl, logits=logits_pl,reduction='weighted_mean')
        loss_op1 = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels_pl, logits=logits_pl,reduction='weighted_sum_by_nonzero_weights')

        # Compute loss
        tf_loss = sess.run(loss_op, feed_dict={onehot_labels_pl: onehot_labels, logits_pl: logits})
        tf_loss1 = sess.run(loss_op1, feed_dict={onehot_labels_pl: onehot_labels, logits_pl: logits})
        np_loss = softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, reduction='mean')

        print(
            "Unit test ({idx:d}/{total:d}), using {nd:d}D arrays/tensors (weights={weight_flag:s}): TF loss(wm) {tf_loss:f},TF loss(nz) {tf_loss1:f} Numpy loss {np_loss:f}"
            .format(idx=test_idx + 1, total=n_tests, nd=len(shape), weight_flag=str(weight_flag), tf_loss=tf_loss,tf_loss1=tf_loss1,
                    np_loss=np_loss))
        assert (np.isclose(tf_loss, np_loss))
        assert (np.isclose(tf_loss1, np_loss))


if __name__ == "__main__":
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.gpu_options.visible_device_list = '0'
    sess = tf.Session(config=config)

    # Do unit testing
    make_unit_tests(n_tests=100, sess=sess)
