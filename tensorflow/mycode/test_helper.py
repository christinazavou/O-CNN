import matplotlib.pyplot as plt
import numpy as np

from tf_layer_utils import *


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
