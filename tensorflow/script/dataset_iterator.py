import tensorflow as tf
import numpy as np


class DataIterator:

    def __init__(self, flags, nout):
        self.flags = flags
        self.data_num = nout

        idxs = np.tile(np.arange(self.data_num), self.flags.rot_num).astype(np.int32)
        rots = np.repeat(np.arange(self.flags.rot_num), self.data_num).astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((idxs, rots)).take(-1)
        # dataset = tf.data.Dataset.from_tensor_slices((np.zeros_like(idxs), np.zeros_like(rots))).take(-1)
        if self.flags.shuffle > 0: dataset = dataset.shuffle(self.data_num * self.flags.rot_num)
        self.itr = dataset.repeat().batch(self.flags.batch_size).prefetch(
            self.flags.batch_size * 2).make_one_shot_iterator()

    def __call__(self, *args, **kwargs):

        return self.itr.get_next()