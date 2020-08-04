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

        self.train_op, self.train_tensors_dict, self.train_summary_op = None, None, None
        self.test_tensors_dict, self.test_summary_op, self.test_summary_placeholder_dict = None, None, None
        self.result_table = None

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
        self.test_summary_op, self.test_summary_placeholder_dict = SummaryDAO \
            .summary_op_for_test(self.test_tensors_dict.keys())
        self.init_logs()

    def init_logs(self):
        self.result_table = PrettyTable(["iter"] + list(self.test_tensors_dict.keys()))

    def update_logs(self, train_iter, test_avg_metrics_dict):
        row = [train_iter]
        for metric_name in self.test_tensors_dict.keys():
            row.append(test_avg_metrics_dict[metric_name])
        self.result_table.add_row(row)

    def run_k_iterations_test(self, session, k):
        avg_results = {key: 0 for key in self.test_tensors_dict.keys()}
        for _ in range(k):
            iter_results = session.run(self.test_tensors_dict)
            for key, result in iter_results.items():
                avg_results[key] += result

        for key, result in avg_results.items():
            avg_results[key] /= k
        return avg_results

    def evaluate(self, session_dao, summary_dao, current_iter):
        print('\nEvaluating on test data ...\n')
        avg_test_metrics_dict = self.run_k_iterations_test(session_dao.session, self.flags.test_iter)
        test_summary = session_dao.session.run(self.test_summary_op,
                                               feed_dict={pl: avg_test_metrics_dict[metric]
                                                          for metric, pl in self.test_summary_placeholder_dict.items()})
        summary_dao.add(test_summary, current_iter)
        self.update_logs(current_iter, avg_test_metrics_dict)
        session_dao.save_iter(current_iter, write_meta_graph=False)
        print(self.result_table)

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
                summary_dao.add(train_summary, i)
                session_dao.update_iter(i)

                if i % self.flags.test_every_iter == 0:
                    self.evaluate(session_dao, summary_dao, i)

            print('Training done!')
