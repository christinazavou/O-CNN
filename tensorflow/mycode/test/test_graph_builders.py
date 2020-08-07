from src.data_parsing import DatasetFactoryDebug
from src.graph_builders import *
from test.helper import mock_train_config


class DatasetTest(tf.test.TestCase):

    def setUp(self):
        self.verificationErrors = []

    def tearDown(self):
        self.assertEqual([], self.verificationErrors)

    def test_convolutions_and_poolings(self):
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


if __name__ == "__main__":
    tf.test.main()
