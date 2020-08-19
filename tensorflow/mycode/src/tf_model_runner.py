import os
import sys

import numpy as np
import tensorflow as tf
from prettytable import PrettyTable
from tqdm import tqdm

sys.path.append("../..")
from libs import octree_property

from src.config import CLASS_TO_LABEL
from src.data_parsing import DatasetFactoryDebug
from src.learning_rate import LRFactory
from src.tf_utils import SummaryDAO, SessionDAO, MisclassifiedOctrees, GraphAccess
from src.visualization import Visualizer


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

        self.train_op, self.lr = None, None
        self.train_output_tensors_dict, self.train_metrics_tensors_dict = None, None
        self.tr_summary_op_always, self.tr_summary_op_occasionally = None, None

        self.test_output_tensors_dict, self.test_metrics_tensors_dict = None, None
        self.test_summary_op, self.test_summary_placeholder_dict = None, None
        self.result_table = None

        # keep tensors of each test example to evaluate each prediction ...
        self.test_octree, self.test_label, self.test_filename = None, None, None

    def init_train_summaries(self):
        train_summaries_always = {'lr': self.lr}
        train_summaries_always.update(self.train_metrics_tensors_dict)
        del train_summaries_always['confusion_matrix']
        train_summaries_occasionally = {'confusion_matrix': self.train_metrics_tensors_dict['confusion_matrix']}
        self.tr_summary_op_always = SummaryDAO.summary_op_for_train(train_summaries_always)
        self.tr_summary_op_occasionally = SummaryDAO.summary_op_for_train(train_summaries_occasionally)

    def build_train_graph(self):
        octree, label, filename = DatasetFactoryDebug(self.train_data_flags)()
        self.train_output_tensors_dict, self.train_metrics_tensors_dict = \
            self.graph_builder(octree, label, self.flags, training=True, reuse=False)
        self.train_op, self.lr = build_solver(self.train_metrics_tensors_dict['loss'], LRFactory(**self.flags))
        self.init_train_summaries()

    def build_test_graph(self, reuse=True):
        self.test_octree, self.test_label, self.test_filename = DatasetFactoryDebug(self.test_data_flags)()
        self.test_output_tensors_dict, self.test_metrics_tensors_dict = \
            self.graph_builder(self.test_octree, self.test_label, self.flags, training=False, reuse=reuse)
        self.test_summary_op, self.test_summary_placeholder_dict = SummaryDAO \
            .summary_op_for_test(self.test_metrics_tensors_dict)
        self.init_test_logs()

    def init_test_logs(self):
        self.result_table = PrettyTable()
        fields = ["iter"]
        for key in self.test_metrics_tensors_dict.keys():
            if key != "confusion_matrix":
                fields.append(key)
        self.result_table.field_names = fields

    def update_logs(self, train_iter, test_avg_metrics_dict):
        row = [train_iter]
        for field_name in self.result_table.field_names[1:]:
            row.append(test_avg_metrics_dict[field_name])
        self.result_table.add_row(row)

    def run_k_iterations_test(self, session, k):
        mo = MisclassifiedOctrees(self.flags.logdir, self.train_data_flags.source_dir)
        avg_results = {key: np.zeros(value.get_shape()) for key, value in self.test_metrics_tensors_dict.items()}
        for _ in range(0, k + 1):
            iter_results, octrees, labels, filenames, outputs = session.run([self.test_metrics_tensors_dict,
                                                                             self.test_octree,
                                                                             self.test_label,
                                                                             self.test_filename,
                                                                             self.test_output_tensors_dict])
            for key, result in iter_results.items():
                avg_results[key] += np.array(result)

            if self.flags.run == 'test':
                misclassified_indices = np.where(labels != outputs['prediction'])[0]
                for mi in misclassified_indices:
                    prediction = outputs['prediction'][mi]
                    probability = outputs['probability'][mi, prediction]
                    if probability > self.flags.misclassified_low_prob:
                        # TODO: find how to parse string from tfrecord without need of decode()
                        mo(filenames[mi].decode('utf-8'), labels[mi], outputs['prediction'][mi], probability)

        for key, result in avg_results.items():
            avg_results[key] /= k
        mo.save()
        return avg_results

    def evaluate_iteration(self, session_dao, summary_dao):
        print('\nEvaluating on test data ...\n')
        avg_test_metrics_dict = self.run_k_iterations_test(session_dao.session, self.flags.test_iter)
        test_summary = session_dao.session.run(self.test_summary_op,
                                               feed_dict={pl: avg_test_metrics_dict[metric]
                                                          for metric, pl in self.test_summary_placeholder_dict.items()})
        if self.flags.run == 'train':
            session_dao.save_iter(session_dao.iter, write_meta_graph=False)
        summary_dao.add(test_summary, session_dao.iter)
        self.update_logs(session_dao.iter, avg_test_metrics_dict)
        print(self.result_table)
        if self.flags.run == 'test':
            Visualizer.confusion_matrix(
                avg_test_metrics_dict['confusion_matrix'].reshape(self.flags.num_class, self.flags.num_class),
                CLASS_TO_LABEL.keys())

    def train_iteration(self, session_dao, summary_dao):
        train_summary, _ = session_dao.session.run([self.tr_summary_op_always, self.train_op])
        summary_dao.add(train_summary, session_dao.iter)

        if session_dao.iter % self.flags.test_every_iter == 0:
            train_summary_confusion = session_dao.session.run(self.tr_summary_op_occasionally)
            summary_dao.add(train_summary_confusion, session_dao.iter)

            self.evaluate_iteration(session_dao, summary_dao)
            self.save_results(os.path.join(session_dao.checkpoints_path, "results_table.txt"))

        session_dao.iter_plus_one()

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
            summary_dao = SummaryDAO(self.flags.logdir, sess.graph)

            print('Start testing ...')
            # todo: pass function in session_dao and let session_dao do the iterations !?
            self.evaluate_iteration(session_dao, summary_dao)
            self.save_results(os.path.join(session_dao.checkpoints_path, "results_table.txt"))
            print('Testing done!')

    def debug(self):
        self.build_test_graph(reuse=False)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        print('Start debugging ...')
        with tf.Session(config=config) as sess:
            session_dao = SessionDAO(sess, self.flags.logdir, keep_max=self.flags.ckpt_num,
                                     load_iter=self.flags.ckpt)
            octrees, labels, filenames = sess.run([self.test_octree, self.test_label, self.test_filename])
            features = sess.run(octree_property(octrees, property_name='feature', depth=0, channel=3, dtype=tf.float32))
            print("features ", features.shape)
            np.savetxt('features_d0_{}.np'.format(self.flags.ckpt), features)
            features = sess.run(octree_property(octrees, property_name='feature', depth=1, channel=3, dtype=tf.float32))
            print("features ", features.shape)
            np.savetxt('features_d1_{}.np'.format(self.flags.ckpt), features)
            features = sess.run(octree_property(octrees, property_name='feature', depth=2, channel=3, dtype=tf.float32))
            print("features ", features.shape)
            np.savetxt('features_d2_{}.np'.format(self.flags.ckpt), features)

        print('Debugging done!')

    def save_results(self, output_path):
        open_mode = 'a'
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
            open_mode = 'w'
        with open(output_path, open_mode) as f:
            f.write(str(self.result_table))

    def amount_of_params(self):
        self.build_train_graph()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            session_dao = SessionDAO(sess, self.flags.logdir, keep_max=self.flags.ckpt_num,
                                     load_iter=self.flags.ckpt)

        GraphAccess.get_total_params(tf.trainable_variables(), verbose=True)

    def run(self):
        eval('self.{}()'.format(self.flags.run))
