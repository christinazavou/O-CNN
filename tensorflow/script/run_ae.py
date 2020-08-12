import os
import tensorflow as tf
from tqdm import tqdm

from config import parse_args
from tfsolver import TFSolver
from dataset import DatasetFactory
from network_ae import make_autoencoder
from ocnn import l2_regularizer


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = parse_args()

# get the autoencoder
autoencoder = make_autoencoder(FLAGS.MODEL)

# define the graph
def compute_graph(dataset='train', training=True, reuse=False):
  flags_data = FLAGS.DATA.train if dataset=='train' else FLAGS.DATA.test
  octree, label = DatasetFactory(flags_data)()
  code, dbe = autoencoder.octree_encoder(octree, training, reuse)
  loss, accu, dbd = autoencoder.octree_decoder(code, octree, training, reuse)
  debug_checks = {'encoder': dbe, 'decoder': dbd}

  with tf.name_scope('total_loss'):
    reg = l2_regularizer('ocnn', FLAGS.LOSS.weight_decay)
    total_loss  = tf.add_n(loss + [reg])
  tensors = loss + [reg] + accu + [total_loss]
  depth = FLAGS.MODEL.depth
  names = ['loss%d' % d for d in range(2, depth + 1)] + ['normal', 'reg'] + \
          ['accu%d' % d for d in range(2, depth + 1)] + ['total_loss']
  return tensors, names, octree, label, debug_checks

# define the solver
class AeTFSolver(TFSolver):
  def __init__(self, flags, compute_graph):
    super(AeTFSolver, self).__init__(flags, compute_graph)

  def decode_shape(self):
    # build graph
    octree, label =  DatasetFactory(FLAGS.DATA.test)()
    code, _ = autoencoder.octree_encoder(octree, training=False, reuse=False)
    octree_pred = autoencoder.octree_decode_shape(code, training=False, reuse=False)

    # checkpoint
    assert(self.flags.ckpt)   # the self.flags.ckpt should be provided
    tf_saver = tf.train.Saver(max_to_keep=20)

    # start
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      # restore and initialize
      self.initialize(sess)
      tf_saver.restore(sess, self.flags.ckpt)
      logdir = self.flags.logdir
      tf.summary.FileWriter(logdir, sess.graph)

      print('Start testing ...')
      if not os.path.exists(os.path.join(logdir, "inputshapes")):
        os.makedirs(os.path.join(logdir, "inputshapes"))
      if not os.path.exists(os.path.join(logdir, "outputshapes")):
        os.makedirs(os.path.join(logdir, "outputshapes"))
      with open(os.path.join(logdir, "input_shapes.txt"), "w") as fi, open(os.path.join(logdir, "output_shapes.txt"), "w") as fo:
        for i in tqdm(range(0, self.flags.test_iter)):
          origin, reconstructed = sess.run([octree, octree_pred])
          with open(os.path.join(logdir, "inputshapes", ('%04d_input.octree' % i)), "wb") as f:
            f.write(origin.tobytes())
          with open(os.path.join(logdir, "outputshapes", ('%04d_output.octree' % i)), "wb") as f:
            f.write(reconstructed.tobytes())
          fi.write(os.path.join(logdir, "inputshapes", ('%04d_input.octree' % i))+"\n")
          fo.write(os.path.join(logdir, "outputshapes", ('%04d_output.octree' % i))+"\n")

# run the experiments
solver = AeTFSolver(FLAGS.SOLVER, compute_graph)
solver.run()
