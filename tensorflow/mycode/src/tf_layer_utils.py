import tensorflow as tf


def make_weights(shape, name='weights'):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05), name=name)


def make_biases(shape, name='biases'):
    return tf.Variable(tf.constant(0.05, shape=shape), name=name)


def convolution_layer(prev_layer, f_size, inp_c, out_c, stride_s):
    _weights = make_weights([f_size, f_size, inp_c, out_c])
    _bias = make_biases([out_c])
    return tf.add(tf.nn.conv2d(prev_layer, _weights, [1, stride_s, stride_s, 1], padding='SAME'), _bias)


def pool_layer(prev_layer, size, stride_s):
    kernel = [1, size, size, 1]
    stride = [1, stride_s, stride_s, 1]
    return tf.nn.max_pool(prev_layer, kernel, stride, padding='SAME')


def activation_layer(prev_layer, type):
    if type == 'relu':
        return tf.nn.relu(prev_layer)
    else:
        raise NotImplemented('unsupported activation type')


def flat_layer(inp):
    input_size = inp.get_shape().as_list()
    if len(input_size) != 4:
        raise NotImplemented('flat layer unsupported for input with dim != 4')
    output_size = input_size[-1] * input_size[-2] * input_size[-3]
    return tf.reshape(inp, [-1, output_size]), output_size


def fc_layer(prev_layer, h_in, h_out):
    _weights = make_weights([h_in, h_out])
    _bias = make_biases([h_out])
    return tf.add(tf.matmul(prev_layer, _weights), _bias)


def dropout_layer(prev_layer, prob):
    return tf.nn.dropout(prev_layer, prob)
