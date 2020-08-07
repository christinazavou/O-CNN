import os

import tensorflow as tf
from numpy import prod


class GraphAccess:

    @staticmethod
    def get_variables(include_substrings=None, exclude_substrings=None, train_only=True, verbose=False):
        vars = tf.trainable_variables() if train_only else tf.all_variables()

        if include_substrings is not None and include_substrings != []:
            vars = [var for var in vars
                    if any([incl_sub for incl_sub in include_substrings
                            if incl_sub.lower() in var.name.lower()])]

        if exclude_substrings is not None and exclude_substrings != []:
            vars = [var for var in vars
                    if not any([exl_sub for exl_sub in exclude_substrings
                                if exl_sub.lower() in var.name.lower()])]

        if verbose:
            print("[*] Variables that include any of {} and exclude any of {} and are trainable:{}:"
                  .format(include_substrings, exclude_substrings, train_only))
            for idx, v in enumerate(vars):
                print("got {}: {} with shape {}".format(idx, v.name, str(v.get_shape())))

        return vars

    @staticmethod
    def get_total_params(variables, exclude_substrings=None, verbose=False):
        total_num = 0
        for idx, var in enumerate(variables):
            name, shape = var.name, var.get_shape()

            exclude = False or exclude_substrings and any([s in name for s in exclude_substrings])
            if not exclude:
                shape_str = '; '.join([str(s) for s in shape])
                shape_num = prod(shape)
                if verbose:
                    print("{:3}, {}, [{}], {}".format(idx, name, shape_str, shape_num))
                total_num += shape_num

        print('Total parameters: {}'.format(total_num))
        return total_num


class Loss:

    # @staticmethod
    # def softmax_cross_entropy(targets, logits, weights, weight_decay=0., scope="softmax_cross_entropy"):
    #     # by default weight_decay is 0 thus no regularization ..
    #     with tf.name_scope(scope):
    #         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
    #         l2regularization = Loss.l2_regularizer(weights, weight_decay)
    #         loss = cost + l2regularization
    #         return cost, l2regularization, loss

    @staticmethod
    def softmax_cross_entropy(label_gt, logit, weights, num_class, weight_decay=0.,
                              scope="softmax_cross_entropy", label_smoothing=0.0):
        with tf.name_scope(scope):
            label_gt = tf.cast(label_gt, tf.int32)
            onehot = tf.one_hot(label_gt, depth=num_class)
            cost = tf.losses.softmax_cross_entropy(onehot, logit, label_smoothing=label_smoothing)
            l2regularization = Loss.l2_regularizer(weights, weight_decay)
            loss = cost + l2regularization
        return cost, l2regularization, loss

    @staticmethod
    def l2_regularizer(weights, weight_decay, scope="l2_regularizer"):
        with tf.name_scope(scope):
            regularizer = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * weight_decay
        return regularizer


class MisclassifiedOctrees:

    def __init__(self, directory):
        self.directory = directory
        self.case_id = 0

    def __call__(self, octree, label, prediction):
        os.makedirs(os.path.join(self.directory, str(self.case_id)))
        self.case_id += 1
        print("label ", label)


class Evaluation:

    # @staticmethod
    # def accuracy(predictions, targets, scope="accuracy"):
    #     with tf.name_scope(scope):
    #         correct_prediction = tf.equal(tf.argmax(targets, 1), tf.argmax(predictions, 1))
    #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     return accuracy

    @staticmethod
    def label_accuracy(label, label_gt):
        label_gt = tf.cast(label_gt, tf.int32)
        accuracy = tf.reduce_mean(tf.to_float(tf.equal(label, label_gt)))
        return accuracy

    @staticmethod
    def classification_metrics(logit, label, num_classes, scope="classification_metrics"):
        with tf.name_scope(scope):
            prediction = tf.argmax(logit, axis=1, output_type=tf.int32)
            accuracy = Evaluation.label_accuracy(prediction, tf.cast(label, tf.int32))
            confusion_matrix = Evaluation.confusion_matrix(prediction, label, num_classes)
        return accuracy, confusion_matrix

    @staticmethod
    def confusion_matrix(prediction, label, num_classes, weights=None):
        matrix = tf.math.confusion_matrix(
            label, prediction, num_classes=num_classes, weights=weights, dtype=tf.dtypes.float32, name=None
        )
        return tf.reshape(matrix, (1, num_classes, num_classes, 1))


class SessionDAO:

    def __init__(self, session, store_dir, keep_max=10, load_iter=None):

        self.session = session
        self.checkpoints_path = os.path.join(store_dir, 'model')
        self.keep_max = keep_max
        self.tf_saver = tf.train.Saver(max_to_keep=keep_max)

        if load_iter is not None and load_iter != "":
            print("Loading iter {}".format(load_iter))
            self.iter = int(load_iter)
            self.load_iter()
        else:
            latest_checkpoint_path = self.get_latest_checkpoint_path()
            if latest_checkpoint_path is not None:
                print("Loading latest_checkpoint_path {}".format(latest_checkpoint_path))
                self.load_session_from_path(latest_checkpoint_path)
                self.iter = SessionDAO.get_iter_from_path(latest_checkpoint_path)
            else:
                print("Setting iter to 1...")
                self.iter = 1

    def iter_plus_one(self):
        self.iter += 1

    def save_iter(self, current_iter=None, write_meta_graph=False):
        if current_iter:
            self.iter = current_iter
        checkpoint_path = os.path.join(self.checkpoints_path, 'iter_%06d.ckpt' % self.iter)
        self.tf_saver.save(self.session, checkpoint_path, write_meta_graph=write_meta_graph)

    def load_iter(self, load_iter=None):
        load_iter = load_iter or self.iter
        checkpoint_path = os.path.join(self.checkpoints_path, 'iter_%06d.ckpt' % load_iter)
        self.load_session_from_path(checkpoint_path)

    def load_session_from_path(self, checkpoint_path):
        self.tf_saver.restore(self.session, checkpoint_path)

    def get_latest_checkpoint_path(self, ):
        # returns path of the latest checkpoint or None if there is no checkpoint under the directory self.checkpoints_path
        return tf.train.latest_checkpoint(self.checkpoints_path)

    def initialize(self):
        print("Initializing session...")
        self.session.run(tf.global_variables_initializer())

    @staticmethod
    def get_iter_from_path(checkpoint_path):
        return int(checkpoint_path[checkpoint_path.find("iter") + 5:-5])


class SummaryDAO:

    def __init__(self, log_dir, graph):
        self.log_dir = log_dir
        self.tf_summary_writer = tf.summary.FileWriter(self.log_dir, graph)

    def add(self, summary, train_iter):
        self.tf_summary_writer.add_summary(summary, train_iter)

    def print(self, event_filename, tag):
        for e in tf.train.summary_iterator(os.path.join(self.log_dir, event_filename)):
            has_value = False
            msg = '{}'.format(e.step)
            for v in e.summary.value:
                if tag in v.tag:
                    msg = msg + ', {}'.format(v.simple_value)
                    has_value = True
            if has_value:
                print(msg)

    @staticmethod
    def summary_op_for_train(tensors_dict):
        # tensors is a dict of name:tensor
        with tf.name_scope("train_summaries"):
            summaries = []
            for tensor_name, tensor in tensors_dict.items():
                if tensor_name == 'confusion_matrix':
                    summaries.append(tf.summary.image(tensor_name, tensor))
                else:
                    summaries.append(tf.summary.scalar(tensor_name, tensor))
            return tf.summary.merge(summaries)

    @staticmethod
    def summary_op_for_test(tensors_dict):
        #   For test we need placeholder tensors ..
        #   because we will feed it with the average summary of k iterations
        with tf.name_scope('test_summaries'):
            summaries = []
            summary_placeholder_dict = {}
            for name in tensors_dict.keys():
                if name == 'confusion_matrix':
                    summary_placeholder_dict[name] = tf.placeholder(tf.float32, tensors_dict[name].get_shape(), name)
                    summaries.append(tf.summary.image(name, summary_placeholder_dict[name]))
                else:
                    summary_placeholder_dict[name] = tf.placeholder(tf.float32, name=name)
                    summaries.append(tf.summary.scalar(name, summary_placeholder_dict[name]))
            return tf.summary.merge(summaries), summary_placeholder_dict
