import tensorflow as tf
import numpy as np

tf.set_random_seed(1)

B, N, C = 4, 100000, 34

# your class weights
class_weights = tf.constant(
    [[[0.0, 1.0, 1.2607750170843908, 1.5005930826950187, 1.086003196497788, 1.3488467965077944, 1.4618322338667538,
       1.3174190951492424, 1.539809107718665, 1.1438476294835398, 1.4151902825998448, 1.5083375754995785,
       1.4857699283179813, 1.5664935071153896, 1.228412737608595, 1.5452717626065522, 1.3300215361581862,
       1.3954368262559722, 1.3967771547392949, 1.3952685623940035, 1.4588113378317014, 1.587808410098552,
       1.4549345122678352, 1.3629362751926624, 1.7781427045873794, 1.5798946403464105, 1.5806155176614685,
       1.6866705953387588, 2.0, 1.659760728786743, 1.757814718996263, 1.8404919947017664, 0.0,
       0.0]]])  # tf.constant([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]])

# logits = tf.random_uniform([B, N, C], minval=-10000, maxval=10000, dtype=tf.float32, name='logit')
# labels = tf.random_uniform([B, N], minval=0, maxval=C+1, dtype=tf.int32, name='target')

# onehot_labels = tf.one_hot(labels, depth=C)
# onehot_labels = tf.cast(onehot_labels, tf.float32)
# weighted_onehot_labels = class_weights * onehot_labels
# # compute your weighted softmax cross entropy loss
# weighted_loss = tf.losses.softmax_cross_entropy(onehot_labels=weighted_onehot_labels, logits=logits)
# loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
# # Create a session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.allow_soft_placement = True
# config.gpu_options.visible_device_list = '0'
# sess = tf.Session(config=config)
# weighted_loss_np, loss_np, weighted_onehot_labels_np, onehot_labels_np = sess.run(
#     [weighted_loss, loss, weighted_onehot_labels, onehot_labels])
# print(weighted_loss_np,loss_np)
#


logits = tf.random_uniform([N, C], minval=-10000, maxval=10000, dtype=tf.float32, name='logit')
labels = tf.random_uniform([N], minval=0, maxval=C+1, dtype=tf.int32, name='target')
class_weights = tf.reshape(class_weights, [C])
# onehot_labels = tf.one_hot(labels, depth=C)
# onehot_labels = tf.cast(onehot_labels, tf.float32)
label_gt = tf.cast(labels, tf.int32)
onehot = tf.one_hot(label_gt, depth=C)
loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot, logits=logits, label_smoothing=0, weights=class_weights)
with tf.Session() as sess:
    print(sess.run(loss))

