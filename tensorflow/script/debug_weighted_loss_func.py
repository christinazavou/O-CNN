import tensorflow as tf
import numpy as np

tf.set_random_seed(1)

B, N, C = 16, 100, 10

logits = tf.random_uniform([B, N, C], minval=-10, maxval=10, dtype=tf.float32, name='logit')
labels = tf.random_uniform([B, N], minval=0, maxval=9, dtype=tf.int32, name='target')

# your class weights
class_weights = tf.constant([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]])
onehot_labels = tf.one_hot(labels, depth=C)
onehot_labels = tf.cast(onehot_labels, tf.float32)
weighted_onehot_labels = class_weights * onehot_labels
# compute your weighted softmax cross entropy loss
weighted_loss = tf.losses.softmax_cross_entropy(onehot_labels=weighted_onehot_labels, logits=logits)
loss=tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)
# Create a session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.visible_device_list = '0'
sess = tf.Session(config=config)
weighted_loss_np,loss_np, weighted_onehot_labels_np, onehot_labels_np = sess.run([weighted_loss,loss, weighted_onehot_labels, onehot_labels])
print()