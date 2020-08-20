import tensorflow as tf

from config import parse_args
from tfsolver import TFSolver
from dataset import DatasetFactory
from network_factory import cls_network
from ocnn import loss_functions

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# configs
FLAGS = parse_args()

# define the graph
class ComputeGraph:
  def __init__(self, flags):
    self.flags = flags

  def __call__(self, dataset='train', training=True, reuse=False, gpu_num=1):
    FLAGS = self.flags
    with tf.device('/cpu:0'):
      flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
      data_iter = DatasetFactory(flags_data)(return_iter=True)

    tower_tensors = []
    for i in range(gpu_num):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('device_%d' % i):
          octree, label = data_iter.get_next()
          logit = cls_network(octree, FLAGS.MODEL, training, reuse)
          losses = loss_functions(logit, label, FLAGS.LOSS.num_class,
                                  FLAGS.LOSS.weight_decay, 'ocnn',
                                  FLAGS.LOSS.label_smoothing)
          losses.append(losses[0] + losses[2])  # total loss
          tower_tensors.append(losses)
          reuse = True

    names = ['loss', 'accu', 'regularizer', 'total_loss']
    tensors = tower_tensors[0] if gpu_num == 1 else list(zip(*tower_tensors))
    return tensors, names, None, None, None


# run the experiments
if __name__ == '__main__':
  solver = TFSolver(FLAGS.SOLVER, compute_graph)
  solver.run()
