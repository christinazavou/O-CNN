import os
import tensorflow as tf
from script.config import parse_args
from script.dataset import DatasetFactory


import sys
from unittest.mock import patch


class DatasetTest(tf.test.TestCase):

    def test_data(self):
        testargs = ['test_dataset.py', "--config", "/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/configs/cls_octree.yaml"]
        with patch.object(sys, 'argv', testargs):
            FLAGS = parse_args()
        octree, label = DatasetFactory(FLAGS.DATA.train)()
        print("octree: ", octree)
        with self.cached_session(use_gpu=True):
            octree_1 = octree.eval()
            print(octree_1, len(octree_1))
        self.assertTrue(octree is not None)


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  tf.test.main()