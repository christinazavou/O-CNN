from unittest import TestCase
import tensorflow as tf
import numpy as np

from ocnn import confusion_matrix


class Test(TestCase):
    def test_confusion_matrix(self):
        prediction = tf.Variable(initial_value=np.array([1,4,5]))
        label = tf.Variable(initial_value=np.array([1,1,5]))
        cm = confusion_matrix(prediction, label, 7)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(cm).shape)
