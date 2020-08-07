from src.data_parsing import DatasetFactoryDebug
from src.graph_builders import *
from test.helper import mock_train_config

class DatasetTest(tf.test.TestCase):

    def setUp(self):
        self.verificationErrors = []

    def tearDown(self):
        self.assertEqual([], self.verificationErrors)

    def test_ocnn_encoder(self):
        flags_data_train = mock_train_config().DATA.train
        depth = 5
        channel = 3

        octree, label, filename = DatasetFactoryDebug(flags_data_train)()

        encoded_data, debug_checks = ocnn_encoder(octree, depth, channel, True, False, True)
        _, debug_checks_fc = ocnn_classification_logit(encoded_data, 40, True, False, True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            debug_checks_shapes, debug_checks_fc_shapes = sess.run([debug_checks, debug_checks_fc])
            print("\ndebug_checks_shapes:\n")
            for key, value in debug_checks_shapes.items():
                print("{}: {}".format(key, value.shape))
            print("\ndebug_checks_fc_shapes:\n")
            for key, value in debug_checks_fc_shapes.items():
                print("{}: {}".format(key, value.shape))
            self.assertEqual(debug_checks_fc_shapes['output shape:'].shape, (32, 40))

        print("test_ocnn_encoder checked")

    def test_convolution_stride(self):
        flags_data_train = mock_train_config().DATA.train

        octree, label, filename = DatasetFactoryDebug(flags_data_train)()
        data = octree_property(octree, property_name="feature", dtype=tf.float32, depth=5, channel=3)
        data = tf.reshape(data, [1, 3, -1, 1])

        with tf.variable_scope(name_or_scope="ena", reuse=False):
            conv_d5_stride1 = octree_conv_bn_relu(data, octree, 5, channels[5], True, stride=1)
        with tf.variable_scope(name_or_scope="dio", reuse=False):
            conv_d5_stride2 = octree_conv_bn_relu(data, octree, 5, channels[5], True, stride=2)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            res1, res2 = sess.run([conv_d5_stride1, conv_d5_stride2])
            print("shape of res1: ", res1.shape)
            print("shape of res2: ", res2.shape)
            self.assertTrue(res1.shape[2] > res2.shape[2])

        print("test_convolution_stride checked")


if __name__ == "__main__":
    tf.test.main()
