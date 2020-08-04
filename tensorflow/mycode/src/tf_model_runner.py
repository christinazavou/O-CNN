import tensorflow as tf
from prettytable import PrettyTable
from tqdm import tqdm

from src.data_parsing import DatasetFactory
from src.learning_rate import LRFactory
from src.tf_utils import SummaryDAO, SessionDAO


def build_solver(total_loss, learning_rate_handle):
    with tf.name_scope('solver'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            global_step = tf.Variable(0, trainable=False, name='global_step')
            lr = learning_rate_handle(global_step)
            solver = tf.train.MomentumOptimizer(lr, 0.9) \
                .minimize(total_loss, global_step=global_step)
    return solver, lr


class TFRunner:

    def __init__(self, train_data_flags, test_data_flags, model_flags, graph_builder):
        self.train_data_flags = train_data_flags
        self.test_data_flags = test_data_flags
        self.flags = model_flags
        self.graph_builder = graph_builder

    def build_train_graph(self):
        octree, label = DatasetFactory(self.train_data_flags)()

        self.train_tensors_dict = self.graph_builder(octree, label, self.flags, training=True, reuse=False)
        self.train_op, lr = build_solver(self.train_tensors_dict['loss'], LRFactory(**self.flags))

        train_summaries = {'lr': lr}
        train_summaries.update(self.train_tensors_dict)
        self.train_summary_op = SummaryDAO.summary_op_for_train(train_summaries)

    def build_test_graph(self):
        octree, label = DatasetFactory(self.test_data_flags)()
        self.test_tensors_dict = self.graph_builder(octree, label, self.flags, training=False, reuse=True)
        self.test_summary_op, self.test_summary_placeholders = SummaryDAO \
            .summary_op_for_test(self.test_tensors_dict.keys())

    def update_logs(self, train_iter, test_avg_metrics):
        if not self.result_table:
            self.result_table = PrettyTable(
                ["iter"] + self.test_tensors_dict.keys()
            )
        self.result_table.add_row([train_iter] + test_avg_metrics)

    def run_k_iterations(self, session, k, tensors):
        num_of_metrics = len(tensors)
        avg_results = [0] * num_of_metrics
        for _ in range(k):
            iter_results = session.run(tensors)
            for metric_idx in range(num_of_metrics):
                avg_results[metric_idx] += iter_results[metric_idx]

        for metric_idx in range(num_of_metrics):
            avg_results[metric_idx] /= k
        return avg_results

    def train(self):
        self.build_train_graph()
        self.build_test_graph()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            session_dao = SessionDAO(sess, self.flags.logdir, keep_max=self.flags.ckpt_num,
                                     load_iter=self.flags.ckpt)
            session_dao.initialize()
            summary_dao = SummaryDAO(self.flags.logdir, sess.graph)

            print('Start training ...')
            for i in tqdm(range(session_dao.iter, self.flags.max_iter + 1), ncols=80):
                train_summary, _ = sess.run([self.train_summary_op, self.train_op])
                summary_dao.add(summary, i)
                session_dao.update_iter(i)

                if i % self.flags.test_every_iter == 0:
                    print('Evaluating on test date ...')
                    avg_test_metrics = self.run_k_iterations(sess, self.flags.test_iter, self.test_tensors_dict)
                    summary = sess.run(self.test_summary_op,
                                       feed_dict=dict(zip(self.test_summary_placeholders, avg_test_metrics)))
                    summary_dao.add(summary, i)
                    self.update_logs(self, i, avg_test_metrics)
                    session_dao.save_iter(i, write_meta_graph=False)

            print('Training done!')
