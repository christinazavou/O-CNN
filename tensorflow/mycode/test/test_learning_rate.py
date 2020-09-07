import numpy as np

from src.learning_rate import *


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


if __name__ == "__main__":
    tf.test.main()
