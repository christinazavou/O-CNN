import sys

import tensorflow as tf

from src.tf_utils import Loss, GraphAccess, Evaluation

sys.path.append("../..")
from libs import octree_full_voxel, octree_max_pool, octree_property, octree_conv_memory, octree_conv_fast


def autoencoder_graph():
    pass


channels = [512, 256, 128, 64, 32, 16, 8, 4, 2]  # convolutions set for up to depth 8


def octree_conv_bn_relu(data, octree, depth, channel, training, kernel_size=[3],
                        stride=1, fast_mode=False):
    with tf.variable_scope('conv_bn_relu'):
        conv_bn = octree_conv_bn(data, octree, depth, channel, training, kernel_size,
                                 stride, fast_mode)
        rl = tf.nn.relu(conv_bn)
    return rl


def octree_conv_bn(data, octree, depth, channel, training, kernel_size=[3],
                   stride=1, fast_mode=False):
    if fast_mode:
        conv = octree_conv_fast(data, octree, depth, channel, kernel_size, stride)
    else:
        conv = octree_conv_memory(data, octree, depth, channel, kernel_size, stride)
    return tf.layers.batch_normalization(conv, axis=1, training=training)


def fc_bn_relu(inputs, nout, training):
    fc = dense(inputs, nout, use_bias=False)
    bn = batch_norm(fc, training)
    return tf.nn.relu(bn)


def dense(inputs, nout, use_bias=False):
    inputs = tf.layers.flatten(inputs)
    fc = tf.layers.dense(inputs, nout, use_bias=use_bias,
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return fc


def batch_norm(inputs, training, axis=1):
    return tf.layers.batch_normalization(inputs, axis=axis, training=training)


def ocnn_encoder(octree, depth, in_channel, training=True, reuse=None, debug=False):
    assert depth > 1 and depth < 9, "Depth {} can't be used.".format(depth)

    debug_checks = {}

    with tf.variable_scope("ocnn_encoder", reuse=reuse):
        data = octree_property(octree, property_name="feature", dtype=tf.float32,
                               depth=depth, channel=in_channel)
        data = tf.reshape(data, [1, in_channel, -1, 1])
        if debug:
            debug_checks['input data reshaped: {}'] = data

        for d in range(depth, 2, -1):
            with tf.variable_scope('depth_%d' % d):
                data = octree_conv_bn_relu(data, octree, d, channels[d], training)
                if debug:
                    debug_checks['output of conv_d{}:'.format(d)] = data
                data, _ = octree_max_pool(data, octree, d)
                if debug:
                    debug_checks['output of pool_d{}:'.format(d)] = data

        with tf.variable_scope("full_voxel"):
            data = octree_full_voxel(data, depth=2)
            if debug:
                debug_checks['output of full_voxel:'] = data
            data = tf.layers.dropout(data, rate=0.5, training=training)
    return data, debug_checks


def ocnn_classification_logit(encoded_data, n_out, training=True, reuse=None, debug=False):
    debug_checks = {}
    with tf.variable_scope("ocnn_classifier", reuse=reuse):
        with tf.variable_scope("fc1"):
            data = fc_bn_relu(encoded_data, channels[2], training=training)
            if debug:
                debug_checks['output of fc_bn_relu:'] = data
            data = tf.layers.dropout(data, rate=0.5, training=training)

        with tf.variable_scope("fc2"):
            _logit = dense(data, n_out, use_bias=True)
            if debug:
                debug_checks['output shape:'] = _logit
    return _logit, debug_checks


def classification_graph(octree, label, flags, training=True, reuse=None):
    encoded_data, _ = ocnn_encoder(octree, flags.depth, flags.channel, training, reuse)
    logit, _ = ocnn_classification_logit(encoded_data, flags.nout, training, reuse)
    # prediction = tf.nn.softmax(logit)  # tf.argmax(logit, axis=1, output_type=tf.int32)

    trainables = GraphAccess.get_variables(None, None, True, True)

    cost, l2reg, loss = Loss.softmax_cross_entropy(label, logit, trainables, flags.num_class,
                                                   weight_decay=flags.weight_decay)
    accuracy = Evaluation.accuracy(logit, label)

    return {'cost': cost, 'l2reg': l2reg, 'loss': loss, 'accuracy': accuracy}
