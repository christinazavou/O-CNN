from src.tf_utils import *
from test.helper import *


class TfRunnerTest(tf.test.TestCase):

    def setUp(self):
        self.verificationErrors = []

    def tearDown(self):
        self.assertEqual([], self.verificationErrors)

    def test_all(self):
        input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        make_graph(input_data, 10)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        try:
            TfRunnerTest.test_solver()
            print("test_solver checked")

            TfRunnerTest.test_total_params()
            print("test_total_params checked")
        except AssertionError as e:
            self.verificationErrors.append(str(e))

    @staticmethod
    def test_solver():
        result = GraphAccess.get_variables_by_name(include_substrings=["Layer"],
                                                   exclude_substrings=["bias", "Embedding"],
                                                   train_only=True,
                                                   verbose=True)
        assert len(result) == 2
        assert result[0].name == 'Layer1/weights:0' or result[1].name == 'Layer1/weights:0'
        assert result[0].name == 'OutLayer/weights:0' or result[1].name == 'OutLayer/weights:0'

    @staticmethod
    def test_total_params():
        total_params = GraphAccess.get_total_params(tf.trainable_variables(), verbose=True)
        assert total_params == 704162


if __name__ == "__main__":
    tf.test.main()
