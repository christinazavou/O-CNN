import io
import shutil
import sys
import tempfile

from src.tf_utils import *
from test.helper import *


class TfUtilsTest(tf.test.TestCase):

    def setUp(self):
        self.verificationErrors = []
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        self.assertEqual([], self.verificationErrors)

    def test_all(self):
        input_data = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        make_graph(input_data, 10)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        try:
            test_solver()
            print("test_solver checked")

            test_total_params()
            print("test_total_params checked")

            session_dao = SessionDAO(self.test_dir, keep_max=2)

            var1 = GraphAccess.get_variables_by_name(['Layer1/weights'])[0]
            min_value_var1 = sess.run(var1).min()
            assert min_value_var1 != 0
            session_dao.save(sess, 100)

            sess.run(var1.assign(np.zeros(var1.get_shape())))
            min_value_var1 = sess.run(var1).min()
            assert min_value_var1 == 0

            session_dao.save(sess, 200)

            assert os.path.exists(os.path.join(self.test_dir, 'model/checkpoint'))
            assert os.path.exists(os.path.join(self.test_dir, 'model/iter_000100.ckpt.data-00000-of-00001'))
            assert os.path.exists(os.path.join(self.test_dir, 'model/iter_000100.ckpt.index'))
            assert os.path.exists(os.path.join(self.test_dir, 'model/iter_000200.ckpt.data-00000-of-00001'))
            assert os.path.exists(os.path.join(self.test_dir, 'model/iter_000200.ckpt.index'))

            sess.run(var1.assign(np.ones(var1.get_shape())))
            min_value_var1 = sess.run(var1).min()
            assert min_value_var1 == 1

            session_dao.load(sess, os.path.join(self.test_dir, 'model/iter_000200.ckpt'))

            min_value_var1 = sess.run(var1).min()
            assert min_value_var1 == 0

        except AssertionError as e:
            self.verificationErrors.append(str(e))


def test_solver():
    result = GraphAccess.get_variables_by_name(include_substrings=["Layer"],
                                               exclude_substrings=["bias", "Embedding"],
                                               train_only=True,
                                               verbose=True)
    assert len(result) == 2
    assert result[0].name == 'Layer1/weights:0' or result[1].name == 'Layer1/weights:0'
    assert result[0].name == 'OutLayer/weights:0' or result[1].name == 'OutLayer/weights:0'


def test_total_params():
    total_params = GraphAccess.get_total_params(tf.trainable_variables(), verbose=True)
    assert total_params == 704162


class TfUtilsSummaryDAOTest(tf.test.TestCase):

    def test_all(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output

        summary_dao = SummaryDAO('resources', None)
        summary_dao.print('events.out.tfevents.1596207515.christina-GE62-7RD', 'cost')

        sys.stdout = sys.__stdout__

        captured_output = captured_output.getvalue()
        for i in range(0, 1000):
            self.assertTrue("{}, ".format(i) in captured_output)
        print("TfUtilsSummaryDAOTest.test_all checked")


if __name__ == "__main__":
    tf.test.main()
