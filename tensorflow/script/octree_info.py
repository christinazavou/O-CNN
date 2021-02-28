import sys

sys.path.append("../..")
from libs import *

from config import parse_args
from dataset_preloader import *
from dataset_iterator import *

from data_augmentation import DataAugmentor


class OctreeInfo:
    def __init__(self, flags, nout, channels):
        self.data = DataLoader()
        self.data = self.data(flags, nout, channels, mem_check=False, test_phase=True)
        self.batch = DataIterator(flags, self.data.tfrecord_num)
        self.data_aug = DataAugmentor(flags)

    def __call__(self):
        pass


if __name__ == "__main__":
    FLAGS = parse_args()
    oct_info = OctreeInfo(FLAGS.DATA.train, FLAGS.MODEL.nout, FLAGS.MODEL.channel)
    sess = tf.Session()
    batch = oct_info.batch()
    for _ in range(oct_info.data.tfrecord_num):
        idx, rot = sess.run(batch)
        print(idx, rot)
        octree, _ = oct_info.data_aug(tf.cast(oct_info.data.points[idx], dtype=tf.float32),
                                      tf.cast(oct_info.data.normals[idx], dtype=tf.float32),
                                      tf.cast(oct_info.data.features[idx], dtype=tf.float32),
                                      oct_info.data.point_labels[idx], rot)
        sess.run(check_octree(octree))
        exit()
