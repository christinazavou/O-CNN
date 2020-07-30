import tensorflow as tf


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
        pass

    # def build_train_graph(self):
    #     self.train_tensors, train_names = self.graph('train', training=True, reuse=False)
    #     self.test_tensors, self.test_names = self.graph('test', training=False, reuse=True)
    #     total_loss = self.train_tensors[train_names.index('total_loss')]  # TODO: use dict
    #     self.train_op, lr = self.build_solver(total_loss, LRFactory(self.flags))
    #     self.summaries(train_names + ['lr'], self.train_tensors + [lr, ], self.test_names)
