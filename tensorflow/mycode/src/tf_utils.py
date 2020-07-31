import os

import tensorflow as tf
from numpy import prod


class GraphAccess:

    @staticmethod
    def get_variables_by_name(include_substrings, exclude_substrings=None, train_only=True, verbose=False):
        t_vars = tf.trainable_variables() if train_only else tf.all_variables()
        d_vars = [var for var in t_vars
                  if any([incl_sub for incl_sub in include_substrings
                          if incl_sub.lower() in var.name.lower()])]

        if exclude_substrings is not None:
            d_vars = [var for var in d_vars
                      if not any([exl_sub for exl_sub in exclude_substrings
                                  if exl_sub.lower() in var.name.lower()])]

        if verbose:
            print("[*] Variables that include any of {} and exclude any of {} and are trainable:{}:"
                  .format(include_substrings, exclude_substrings, train_only))
            for idx, v in enumerate(d_vars):
                print("got {}: {} with shape {}".format(idx, v.name, str(v.get_shape())))

        return d_vars

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

    @staticmethod
    def softmax_cross_entropy(targets, logits, weights, weight_decay=0.):
        # by default weight_decay is 0 thus no regularization ..
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        l2regularization = Loss.l2_regularizer(weights, weight_decay)
        loss = cost + l2regularization
        return cost, l2regularization, loss

    @staticmethod
    def l2_regularizer(weights, weight_decay):
        with tf.name_scope('l2_regularizer'):
            regularizer = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * weight_decay
        return regularizer


class SessionDAO:

    def __init__(self, folder, keep_max=10):
        self.ckpt_path = os.path.join(folder, 'model')
        self.keep_max = keep_max
        self.tf_saver = tf.train.Saver(max_to_keep=keep_max)

    def save(self, session, ckpt, write_meta_graph=False):
        ckpt_name = os.path.join(self.ckpt_path, 'iter_%06d.ckpt' % ckpt)
        self.tf_saver.save(session, ckpt_name, write_meta_graph=write_meta_graph)

    def load(self, session, ckpt):
        self.tf_saver.restore(session, ckpt)
