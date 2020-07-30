import matplotlib.pyplot as plt
import numpy as np
from yacs.config import CfgNode as CN

from learning_rate import *
from tf_layer_utils import *
from tf_model_runner import *
from tf_utils import *


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


def make_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    image_index = 7777
    plt.imshow(x_train[image_index], cmap='Greys')
    plt.title("label {}".format(y_train[image_index]))
    plt.show()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train = np.divide(x_train, 255)
    x_test = np.divide(x_test, 255)

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    print("x_train ", x_train.shape, " x_test ", x_test.shape)
    print("y_train ", y_train.shape, " y_test ", y_test.shape)
    return x_train, y_train, x_test, y_test


def make_graph(image, num_classes):
    with tf.variable_scope('Layer1'):
        l1 = convolution_layer(image, 3, 1, 28, 1)
        l1 = pool_layer(l1, 2, 2)
        l1 = activation_layer(l1, 'relu')
        print("layer 1:", l1.get_shape())

    with tf.variable_scope('EmbeddingLayer'):
        embeddings, embeddings_length = flat_layer(l1)
        print("embeddings: ", embeddings.get_shape())

        h1 = fc_layer(embeddings, embeddings_length, 128)
        h1 = activation_layer(h1, 'relu')
        h1 = dropout_layer(h1, 0.2)
        print("h1: ", h1.get_shape())

    with tf.variable_scope('OutLayer'):
        o = fc_layer(h1, 128, num_classes)
        return o


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
