import tensorflow as tf

X = {
    1: tf.get_variable('X1', shape=(1, 128, 10, 3), dtype=tf.float32, initializer=tf.glorot_normal_initializer()),
    2: tf.get_variable('X2', shape=(1, 128, 10, 1), dtype=tf.float32, initializer=tf.glorot_normal_initializer()),
    3: tf.get_variable('X3', shape=(1, 128, 128, 1), dtype=tf.float32, initializer=tf.glorot_normal_initializer()),
    4: tf.get_variable('X4', shape=(1, 128, 10, 10), dtype=tf.float32, initializer=tf.glorot_normal_initializer()),
}
CONV = {
    1: tf.layers.conv2d(X[1], 1, (3, 1)),
    2: tf.layers.conv2d(X[2], 1, (3, 1)),
    3: tf.layers.conv2d(X[3], 1, (3, 3)),
    4: tf.layers.conv2d(X[2], 1, (8, 1), strides=(8, 1)),
    5: tf.layers.conv2d(X[2], 32, (8, 1), strides=(8, 1)),
    6: tf.layers.conv2d(X[2], 32, (8, 1), strides=(8, 1), data_format='channels_first'),
    7: tf.layers.conv2d(X[4], 32, (8, 8), strides=(8, 8), data_format='channels_first'),
    8: tf.layers.conv2d(X[4], 32, (5, 5), strides=(1, 1), data_format='channels_first'),
}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    OUT = sess.run(CONV)
    assert OUT[1].shape == (1, 126, 10, 1)
    assert OUT[2].shape == OUT[1].shape
    assert OUT[3].shape == (1, 126, 126, 1)
    assert OUT[4].shape == (1, 16, 10, 1)
    assert OUT[5].shape == (1, 16, 10, 32)
    assert OUT[6].shape == (1, 32, 1, 1)
    assert OUT[7].shape == (1, 32, 1, 1)
    assert OUT[8].shape == (1, 32, 6, 6)
