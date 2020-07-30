from tf_layer_utils import *
from tf_utils import *


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


class TfRunnerTest(tf.test.TestCase):

    def test_solver(self):
        input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

        logits = make_graph(input_data, 10)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        result = get_variables_by_name(include_substrings=["Layer"],
                                       exclude_substrings=["bias", "Embedding"],
                                       train_only=True,
                                       verbose=True)
        self.assertEqual(len(result), 2)
        self.assertTrue(result[0].name == 'Layer1/weights:0' or result[1].name == 'Layer1/weights:0')
        self.assertTrue(result[0].name == 'OutLayer/weights:0' or result[1].name == 'OutLayer/weights:0')


if __name__ == "__main__":
    tf.test.main()
