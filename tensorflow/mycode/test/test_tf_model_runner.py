from yacs.config import CfgNode as CN

from src.learning_rate import *
from src.tf_model_runner import *
from src.tf_utils import GraphAccess, Loss, SummaryDAO
from test.helper import *


def make_flags():
    flags = CN()
    flags.max_iter = 1000  # Maximum training iterations
    flags.lr_type = 'step'  # Learning rate type: step or cos
    flags.learning_rate = 0.001  # Initial learning rate
    flags.gamma = 0.1  # Learning rate step-wise decay
    flags.step_size = (1000,)  # Learning rate step size.
    flags.batch_size = 128
    flags.display = 10
    flags.weight_decay = 0.0005
    return flags


def make_accuracy(targets, predictions):
    correct_prediction = tf.equal(tf.argmax(targets, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


class TfRunnerTest(tf.test.TestCase):

    def test_solver(self):
        flags = make_flags()
        x_train, y_train, x_test, y_test = make_data()

        input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        target_data = tf.placeholder(tf.float32, shape=[None, 10])

        logits = make_graph(input_data, 10)
        predictions = tf.nn.softmax(logits)

        trainables = GraphAccess.get_variables_by_name('layer', train_only=True, verbose=True)
        cost, l2reg, loss = Loss.softmax_cross_entropy(target_data, logits, trainables,
                                                       flags.weight_decay)
        accuracy = make_accuracy(target_data, predictions)

        train_op, lr = build_solver(cost, LRFactory(**flags))

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        summary_dao = SummaryDAO('tmp', sess.graph)
        train_summary_op = summary_dao.summary_op('train_summaries', {'cost': cost, 'l2reg': l2reg, 'loss': loss})
        test_summary_op = summary_dao.summary_op('test_summaries', {'cost': cost, 'l2reg': l2reg, 'loss': loss})

        for i in range(flags.max_iter):

            images = x_train[i * flags.batch_size: (i + 1) * flags.batch_size]
            labels = y_train[i * flags.batch_size: (i + 1) * flags.batch_size]

            train_cost = sess.run(cost, feed_dict={input_data: images, target_data: labels})
            print('train_cost', train_cost)

            _, train_summaries = sess.run([train_op, train_summary_op],
                                          feed_dict={input_data: images, target_data: labels})
            summary_dao.add(train_summaries, i)

            if np.mod(i, flags.display) == 0:
                test_cost, test_acc, test_summaries = sess.run([cost, accuracy, test_summary_op],
                                                               feed_dict={input_data: x_test, target_data: y_test})
                print('iter number ', i, "test cost =", "{:.3f}".format(test_cost),
                      "test accuracy: {:.3f}".format(test_acc))
                summary_dao.add(test_summaries, i)

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
