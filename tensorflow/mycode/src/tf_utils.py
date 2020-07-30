import tensorflow as tf


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
