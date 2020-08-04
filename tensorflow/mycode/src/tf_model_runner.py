import tensorflow as tf
from prettytable import PrettyTable

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

    def __init__(self, flags, graph_builder):
        self.train_data_flags = flags.data.train
        self.test_data_flags = flags.data.test
        self.flags = flags.train

    def build_train_graph(self):
        octree, label = DatasetFactory(self.train_data_flags)()

        self.train_tensors = self.graph_builder(octree, label, training=True, reuse=False)
        self.train_op, lr = build_solver(self.train_tensors['loss'], LRFactory(**self.flags))

        train_summaries = {'lr': lr}
        train_summaries.update(self.train_tensors)
        self.train_summary_op = SummaryDAO.summary_op('train_summaries', train_summaries)

    def update_logs(self):
        self.result_table = PrettyTable(
            ["iter", "lr", "cost", "l2reg", "loss", "accuracy"]
        )
        self.result_table.add_row(
            [5, 10, 15]
        )

    def build_test_graph(self):
        octree, label = DatasetFactory(self.test_data_flags)()
        self.test_tensors = self.graph_builder(octree, label, training=False, reuse=True)
        self.test_summary_op = SummaryDAO.summary_op('test_summaries', self.test_tensors)

    def train(self):
        self.build_train_graph()

        session_dao = SessionDAO(self.flags.logdir, keep_max=self.flags.ckpt_num)

        start_iter = 1

        if self.flags.ckpt:
            ckpt = self.flags.ckpt
        else:
            ckpt = tf.train.latest_checkpoint(ckpt_path)
            if ckpt: start_iter = int(ckpt[ckpt.find("iter") + 5:-5]) + 1

        # session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)

            print('Initialize ...')
            self.initialize(sess)
            if ckpt: self.restore(sess, ckpt)

            print('Start training ...')
            for i in tqdm(range(start_iter, self.flags.max_iter + 1), ncols=80):
                # training
                summary, _ = sess.run([self.summ_train, self.train_op])
                summary_writer.add_summary(summary, i)

                # testing
                if i % self.flags.test_every_iter == 0:
                    # run testing average
                    avg_test = self.run_k_iterations(sess, self.flags.test_iter, self.test_tensors)

                    # run testing summary
                    summary = sess.run(self.summ_test,
                                       feed_dict=dict(zip(self.summ_holder, avg_test)))
                    summary_writer.add_summary(summary, i)
                    self.summ2txt(avg_test, i)

                    # save session
                    ckpt_name = os.path.join(ckpt_path, 'iter_%06d.ckpt' % i)
                    self.tf_saver.save(sess, ckpt_name, write_meta_graph=False)

            print('Training done!')
