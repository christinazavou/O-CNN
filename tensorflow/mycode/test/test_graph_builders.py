from src.data_parsing import DatasetFactory
from test.helper import mock_config
from src.graph_builders import *


class DatasetTest(tf.test.TestCase):

    def setUp(self):
        self.verificationErrors = []

    def tearDown(self):
        self.assertEqual([], self.verificationErrors)

    def test_convolutions_and_poolings(self):
        flags_data_train = mock_config().DATA.train
        depth = 5
        channel = 3
        channels = [512, 256, 128, 64, 32, 16, 8, 4, 2]
        training= True

        octree, label = DatasetFactory(flags_data_train)()

        data_check = {}

        with tf.variable_scope("ocnn_encoder", reuse=False):
            data = octree_property(octree, property_name="feature", dtype=tf.float32,
                                   depth=depth, channel=channel)
            data = tf.reshape(data, [1, channel, -1, 1])
            data_check['feature_reshaped'] = data

            for d in range(depth, 2, -1):
                with tf.variable_scope('depth_%d' % d):
                    data = octree_conv_bn_relu(data, octree, d, channels[d], training)
                    data_check['conv_depth{}'.format(d)] = data
                    data, _ = octree_max_pool(data, octree, d)
                    data_check['pool_depth{}'.format(d)] = data

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            data_check_results = sess.run(data_check)
            print("data_check_results:", {key: value.shape for key, value in data_check_results.items()})


if __name__ == "__main__":
    tf.test.main()
