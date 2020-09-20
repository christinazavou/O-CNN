import unittest

import numpy as np

from learning_rate import *


def make_flags():
    from yacs.config import CfgNode as CN

    flags = CN()
    flags.max_iter = 160000  # Maximum training iterations
    flags.lr_type = 'step'  # Learning rate type: step or cos
    flags.learning_rate = 0.1  # Initial learning rate
    flags.gamma = 0.1  # Learning rate step-wise decay
    flags.step_size = (400,)  # Learning rate step size.
    return flags


class DatasetTest(tf.test.TestCase):

    def test_lr(self):
        flags = make_flags()
        global_step = tf.Variable(0, trainable=False)
        lr_factory = LRFactory(**flags)
        self.assertTrue(isinstance(lr_factory.lr, StepLR))
        self.assertEqual(lr_factory.lr.gamma, flags.gamma)
        self.assertEqual(lr_factory.lr.step_size, flags.step_size)
        self.assertEqual(lr_factory.lr.learning_rate, flags.learning_rate)
        lr_step = lr_factory(global_step)

        with tf.Session() as sess:
            lr_current = sess.run(lr_step, feed_dict={global_step: 0})
            self.assertEqual(np.float16(lr_current), np.float16(0.1))
            lr_current = sess.run(lr_step, feed_dict={global_step: 400})
            self.assertEqual(np.float16(lr_current), np.float16(0.1))

            lr_current = sess.run(lr_step, feed_dict={global_step: 401})
            self.assertEqual(np.float16(lr_current), np.float16(0.01))
            lr_current = sess.run(lr_step, feed_dict={global_step: 800})
            self.assertEqual(np.float16(lr_current), np.float16(0.01))

            lr_current = sess.run(lr_step, feed_dict={global_step: 801})
            self.assertEqual(np.float16(lr_current), np.float16(0.001))

            lr_current = sess.run(lr_step, feed_dict={global_step: 5 * 400})
            self.assertEqual(np.float16(lr_current), np.float16(0.00001))

            lr_current = sess.run(lr_step, feed_dict={global_step: 6 * 400})
            self.assertEqual(np.float16(lr_current), np.float16(0.000001))
            lr_current = sess.run(lr_step, feed_dict={global_step: 7 * 400})
            self.assertEqual(np.float16(lr_current), np.float16(0.000001))


class test_cond(unittest.TestCase):

    def test_cond1(self):
        a = tf.constant(1)
        b = tf.constant(-1)
        res = tf.cond(tf.greater(a, b), true_fn=lambda: a, false_fn=lambda: b)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(res)
            assert result == 1

    def test_cond2(self):
        a = tf.constant(1)
        b = tf.constant(-1)

        with self.assertRaises(Exception) as context:
            res = tf.cond(2 > 1, true_fn=lambda: a, false_fn=lambda: b)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(res)
        self.assertTrue("pred must not be a Python bool" in str(context.exception))

    def test_cond3(self):
        a = tf.constant(1)
        b = -1

        res = tf.cond(tf.greater(a, b), true_fn=lambda: a, false_fn=lambda: b)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(res)
            self.assertEqual(result, 1)

    def test_cond4(self):
        a = tf.Variable(initial_value=0)
        b = -1

        a = tf.cond(tf.greater(a, b), true_fn=lambda: 10, false_fn=lambda: -10)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(a)
            self.assertEqual(result, 10)

    def test_cond5(self):
        a = tf.Variable(initial_value=0)
        b = -1

        a, b = tf.cond(tf.greater(a, b), true_fn=lambda: [-10, -20], false_fn=lambda: [10, 10])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            resulta, resultb = sess.run([a, b])
            self.assertEqual(resulta, -10)
            self.assertEqual(resultb, -20)

    def test_assign_instead_doesnt_work(self):
        a = tf.Variable(initial_value=0)
        b = -1

        tf.cond(tf.greater(a, b),
                true_fn=lambda: tf.assign(a, 10),
                false_fn=lambda: tf.assign(a, 20))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            resulta = sess.run(a)
            self.assertEqual(resulta, 0)


def make_plateau_flags():
    from yacs.config import CfgNode as CN

    flags = CN()
    flags.mode = 'min'
    flags.lr_type = 'plateau'  # Learning rate type: step or cos
    flags.learning_rate = 0.1  # Initial learning rate
    flags.patience = 0
    flags.threshold_mode = 'rel'
    flags.threshold = 1e-4
    flags.cooldown = 0  # ####################
    flags.min_lr = 0.0
    flags.eps = 1e-8
    flags.gamma = 0.1
    return flags


class OnPlateauLrTest(tf.test.TestCase):

    def test_lr_is_not_working(self):
        flags = make_plateau_flags()
        lr_metric = OnPlateauLR(flags)
        curr_metric = tf.Variable(0.2, trainable=False)
        lr_value = lr_metric(curr_metric)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            lr_current1 = sess.run(lr_value, feed_dict={curr_metric: 0.2})
            lr_current2 = sess.run(lr_value, feed_dict={curr_metric: 0.4})
            lr_current3 = sess.run(lr_value, feed_dict={curr_metric: 0.6})
            lr_current4 = sess.run(lr_value, feed_dict={curr_metric: 0.1})
            assert (lr_current1 == lr_current2 == lr_current3 == lr_current4) and (0.10 < lr_current1 < 0.11)

    def test_lrpy_is_ok(self):
        flags = make_plateau_flags()
        lr_metric = OnPlateauLRPy(flags)

        lr_current1 = lr_metric(0.2)
        lr_current2 = lr_metric(0.4)
        lr_current3 = lr_metric(0.6)
        lr_current4 = lr_metric(0.1)
        assert not (lr_current1 == lr_current2 == lr_current3 == lr_current4)


if __name__ == "__main__":
    tf.test.main()
