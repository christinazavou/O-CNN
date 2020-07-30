from test_helper import *
from tf_utils import *


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
