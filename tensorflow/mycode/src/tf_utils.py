import tensorflow as tf


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
