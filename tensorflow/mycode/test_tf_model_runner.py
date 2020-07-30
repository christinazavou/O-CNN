from yacs.config import CfgNode as CN

from learning_rate import *
from test_helper import *
from tf_model_runner import *


def make_flags():
    flags = CN()
    flags.max_iter = 1000  # Maximum training iterations
    flags.lr_type = 'step'  # Learning rate type: step or cos
    flags.learning_rate = 0.001  # Initial learning rate
    flags.gamma = 0.1  # Learning rate step-wise decay
    flags.step_size = (1000,)  # Learning rate step size.
    flags.batch_size = 128
    flags.display = 10
    return flags


# def make_loss(target_data, predicted_data, weights):
#     cost = tf.reduce_sum(tf.square(target_data - predicted_data))
#
#     l2regularization = tf.reduce_sum(tf.square(weights[0]))
#     for w in weights[1:]:
#         l2regularization += tf.reduce_sum(tf.square(w))
#
#     loss = cost + l2regularization * 0.00005
#     return loss, cost


class TfRunnerTest(tf.test.TestCase):

    def test_solver(self):
        flags = make_flags()
        x_train, y_train, x_test, y_test = make_data()

        input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        target_data = tf.placeholder(tf.float32, shape=[None, 10])

        logits = make_graph(input_data, 10)
        predictions = tf.nn.softmax(logits)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target_data))
        correct_prediction = tf.equal(tf.argmax(target_data, 1), tf.argmax(predictions, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_op, lr = build_solver(cost, LRFactory(**flags))

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        for i in range(flags.max_iter):

            images = x_train[i * flags.batch_size: (i + 1) * flags.batch_size]
            labels = y_train[i * flags.batch_size: (i + 1) * flags.batch_size]

            train_cost = sess.run(cost, feed_dict={input_data: images, target_data: labels})
            print('train_cost', train_cost)

            sess.run(train_op, feed_dict={input_data: images, target_data: labels})

            if np.mod(i, flags.display) == 0:
                test_cost, test_acc = sess.run([cost, accuracy], feed_dict={input_data: x_test, target_data: y_test})
                print('iter number ', i, "test cost =", "{:.3f}".format(test_cost),
                      "test accuracy: {:.3f}".format(test_acc))

        # _, all_weights_value1, lr_value1 = sess.run([train_op, all_weights, lr],
        #                                             feed_dict={input_data: x_train,
        #                                                        target_data: y_train})
        # 
        # _, all_weights_value2, lr_value2 = sess.run([train_op, all_weights, lr],
        #                                             feed_dict={input_data: x_train,
        #                                                        target_data: y_train})
        # 
        # for w1, w2 in zip(all_weights_value1, all_weights_value2):
        #     self.assertEqual(w1.shape, w2.shape)
        #     self.assertFalse(np.all(np.equal(w1, w2)))


if __name__ == "__main__":
    tf.test.main()
