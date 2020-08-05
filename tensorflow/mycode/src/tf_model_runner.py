import os

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

    def build_test_graph(self, reuse=True):
        octree, label = DatasetFactory(self.test_data_flags)()
        self.test_tensors_dict = self.graph_builder(octree, label, self.flags, training=False, reuse=reuse)
        self.test_summary_op, self.test_summary_placeholder_dict = SummaryDAO \
            .summary_op_for_test(list(self.test_tensors_dict.keys()))
        self.init_logs()

    def init_logs(self):
        self.result_table = PrettyTable()
        self.result_table.field_names = ["iter"] + list(self.test_tensors_dict.keys())

    def update_logs(self, train_iter, test_avg_metrics_dict):
        row = [train_iter]
        for field_name in self.result_table.field_names[1:]:
            row.append(test_avg_metrics_dict[field_name])
        self.result_table.add_row(row)

    def run_k_iterations_test(self, session, k):
        avg_results = {key: 0 for key in self.test_tensors_dict.keys()}
        for _ in range(0, k + 1):
            iter_results = session.run(self.test_tensors_dict)
            for key, result in iter_results.items():
                avg_results[key] += result

        for key, result in avg_results.items():
            avg_results[key] /= k
        return avg_results

    def evaluate_iteration(self, session_dao, summary_dao, save=True):
        print('\nEvaluating on test data ...\n')
        avg_test_metrics_dict = self.run_k_iterations_test(session_dao.session, self.flags.test_iter)
        test_summary = session_dao.session.run(self.test_summary_op,
                                               feed_dict={pl: avg_test_metrics_dict[metric]
                                                          for metric, pl in self.test_summary_placeholder_dict.items()})
        if save:
            session_dao.save_iter(session_dao.iter, write_meta_graph=False)
        summary_dao.add(test_summary, session_dao.iter)
        self.update_logs(session_dao.iter, avg_test_metrics_dict)
        print(self.result_table)

    def train_iteration(self, session_dao, summary_dao):
        train_summary, _ = session_dao.session.run([self.train_summary_op, self.train_op])
        summary_dao.add(train_summary, session_dao.iter)
        session_dao.iter_plus_one()

        if session_dao.iter % self.flags.test_every_iter == 0:
            self.evaluate_iteration(session_dao, summary_dao)
            self.save_results(os.path.join(session_dao.checkpoints_path, "results_table.txt"))

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
            # todo: pass function in session_dao and let session_dao do the iterations !?
            for _ in tqdm(range(session_dao.iter, self.flags.max_iter + 1), ncols=80):
                self.train_iteration(session_dao, summary_dao)
            print('Training done!')

    def test(self):
        self.build_test_graph(reuse=False)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            session_dao = SessionDAO(sess, self.flags.logdir, keep_max=self.flags.ckpt_num,
                                     load_iter=self.flags.ckpt)
            session_dao.initialize()
            summary_dao = SummaryDAO(self.flags.logdir, sess.graph)

            print('Start testing ...')
            # todo: pass function in session_dao and let session_dao do the iterations !?
            self.evaluate_iteration(session_dao, summary_dao, save=False)
            self.save_results(os.path.join(session_dao.checkpoints_path, "results_table.txt"))
            print('Testing done!')

    def save_results(self, output_path):
        open_mode = 'a'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
            open_mode = 'w'
        with open(output_path, open_mode) as f:
            f.write(str(self.result_table))

    def run(self):
        eval('self.{}()'.format(self.flags.run))
