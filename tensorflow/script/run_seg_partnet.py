import os

import tensorflow as tf
import numpy as np

from config import parse_args, FLAGS
from partnet_labels import find_category, LEVEL3_LABELS, LEVEL3_COLORS, decimal_to_rgb
from tfsolver import TFSolver
from network_factory import seg_network
from dataset import DatasetFactory
from ocnn import loss_functions_seg, build_solver, get_seg_label, loss_functions_seg_debug_checks
from libs import points_property, octree_property, octree_decode_key

# Add config
FLAGS.LOSS.point_wise = True


# get the label and pts
def get_point_info(points, mask_ratio=0, mask=-1):
  with tf.name_scope('points_info'):
    pts   = points_property(points, property_name='xyz', channel=4)
    label = points_property(points, property_name='label', channel=1)
    label = tf.reshape(label, [-1])
    label_mask = label > mask  # mask out invalid points, -1
    if mask_ratio > 0:         # random drop some points to speed up training
      rnd_mask = tf.random.uniform(tf.shape(label_mask)) > mask_ratio
      label_mask = tf.logical_and(label_mask, rnd_mask)
    pts   = tf.boolean_mask(pts, label_mask)
    label = tf.boolean_mask(label, label_mask)
  return pts, label


# IoU
def tf_IoU_per_shape(pred, label, class_num, mask=-1):
  with tf.name_scope('IoU'):
    # Set mask to 0 to filter unlabeled points, whose label is 0
    label_mask = label > mask  # mask out label
    pred = tf.boolean_mask(pred, label_mask)
    label = tf.boolean_mask(label, label_mask)
    pred = tf.argmax(pred, axis=1, output_type=tf.int32)

    intsc, union = [None] * class_num, [None] * class_num
    for k in range(class_num):
      pk, lk = tf.equal(pred, k), tf.equal(label, k)
      intsc[k] = tf.reduce_sum(tf.cast(pk & lk, dtype=tf.float32))
      union[k] = tf.reduce_sum(tf.cast(pk | lk, dtype=tf.float32))
  return intsc, union


# define the graph
class ComputeGraphSeg:
  def __init__(self, flags):
    self.flags = flags

  def create_dataset(self, flags_data):
    return DatasetFactory(flags_data)(return_iter=True)

  def __call__(self, dataset='train', training=True, reuse=False, gpu_num=1):

    debug_checks = {}

    FLAGS = self.flags
    with tf.device('/cpu:0'):
      flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
      data_iter = self.create_dataset(flags_data)

    tower_tensors = []
    for i in range(gpu_num):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('device_%d' % i):
          octree, _labels, points = data_iter.get_next()
          debug_checks["{}/input_octree".format(dataset)] = octree
          debug_checks["{}/input_points".format(dataset)] = points
          debug_checks["{}/input_labels".format(dataset)] = _labels
          pts, label = get_point_info(points, flags_data.mask_ratio)
          print("mask ratio for {} is {}".format(dataset, flags_data.mask_ratio))
          debug_checks["{}/input_point_info/points".format(dataset)] = pts
          debug_checks["{}/input_point_info/labels".format(dataset)] = label
          debug_checks["{}/input_point_info/normals".format(dataset)] = points_property(
            points, property_name='normal', channel=3)
          if not FLAGS.LOSS.point_wise:
            pts, label = None, get_seg_label(octree, FLAGS.MODEL.depth_out)
            debug_checks["{}/input_seg_label/points"] = pts
            debug_checks["{}/input_seg_label/label"] = label
          logit = seg_network(octree, FLAGS.MODEL, training, reuse, pts=pts)
          debug_checks["{}/logit".format(dataset)] = logit
          losses, dc = loss_functions_seg_debug_checks(logit, label, FLAGS.LOSS.num_class,
                                      FLAGS.LOSS.weight_decay, 'ocnn', mask=0)
          debug_checks.update(dc)
          tensors = losses + [losses[0] + losses[2]]  # total loss
          names = ['loss', 'accu', 'regularizer', 'total_loss']

          if flags_data.batch_size == 1:
            num_class = FLAGS.LOSS.num_class
            intsc, union = tf_IoU_per_shape(logit, label, num_class, mask=0)
            iou = tf.constant(0.0)     # placeholder, calc its value later
            tensors = [iou] + tensors + intsc + union
            names = ['iou'] + names + \
                    ['intsc_%d' % i for i in range(num_class)] + \
                    ['union_%d' % i for i in range(num_class)]

          tower_tensors.append(tensors)
          reuse = True

    tensors = tower_tensors[0] if gpu_num == 1 else list(zip(*tower_tensors))
    return tensors, names, debug_checks


# define the solver
class PartNetSolver(TFSolver):
  def __init__(self, flags, compute_graph,  build_solver=build_solver):
    super(PartNetSolver, self).__init__(flags.SOLVER, compute_graph, build_solver)
    self.num_class = flags.LOSS.num_class # used to calculate the IoU

  def result_callback(self, avg_results):
    # calc part-IoU, update `iou`, this is in correspondence with Line 77
    iou_avg = 0.0
    ious = [0] * self.num_class
    for i in range(1, self.num_class):  # !!! Ignore the first label
      instc_i = avg_results[self.test_names.index('intsc_%d' % i)]
      union_i = avg_results[self.test_names.index('union_%d' % i)]
      ious[i] = instc_i / (union_i + 1.0e-10)
      iou_avg = iou_avg + ious[i]
    iou_avg = iou_avg / (self.num_class - 1)
    avg_results[self.test_names.index('iou')] = iou_avg
    return avg_results

  def test(self):
    # build graph
    self.build_test_graph()

    # checkpoint
    assert(self.flags.ckpt)   # the self.flags.ckpt should be provided
    tf_saver = tf.train.Saver(max_to_keep=10)

    # start
    num_tensors = len(self.test_tensors)
    avg_test = [0] * num_tensors
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)
      self.summ2txt(self.test_names, 'batch')

      # restore and initialize
      self.initialize(sess)
      print('Restore from checkpoint: %s' % self.flags.ckpt)
      tf_saver.restore(sess, self.flags.ckpt)

      category = find_category(self.flags.ckpt)
      assert category is not None
      predicted_ply_dir = os.path.join(self.flags.logdir, "predicted_ply_{}".format(category))
      if not os.path.exists(predicted_ply_dir):
        os.makedirs(predicted_ply_dir)

      print('Start testing ...')
      for i in range(0, self.flags.test_iter):
        iter_test_result, iter_test_dc = sess.run([self.test_tensors, self.debug_test_checks])

        points, labels, predictions, normals = iter_test_dc['test/input_point_info/points'], \
                                               iter_test_dc['test/input_point_info/labels'], \
                                               iter_test_dc['softmax_loss/prediction'], \
                                               iter_test_dc["test/input_point_info/normals"]

        if predictions.shape[0] != points.shape[0]:
          continue

        iter_test_result = self.result_callback(iter_test_result)

        dec_colors = LEVEL3_COLORS[category]
        cp = np.array([decimal_to_rgb(dec_colors[p]) for p in predictions])
        save_ply(os.path.join(predicted_ply_dir, "{}.ply".format(i)), points[:, 0:3], normals, cp)

        # run testing average
        for j in range(num_tensors):
          avg_test[j] += iter_test_result[j]
        # print the results
        reports = 'batch: %04d; ' % i
        for j in range(num_tensors):
          reports += '%s: %0.4f; ' % (self.test_names[j], iter_test_result[j])
        print(reports)
        self.summ2txt(iter_test_result, i)

    # Final testing results
    for j in range(num_tensors):
      avg_test[j] /= self.flags.test_iter
    avg_test = self.result_callback(avg_test)
    # print the results
    print('Testing done!\n')
    reports = 'ALL: %04d; ' % self.flags.test_iter
    for j in range(num_tensors):
      reports += '%s: %0.4f; ' % (self.test_names[j], avg_test[j])
    print(reports)
    self.summ2txt(avg_test, 'ALL')


def save_ply(filename, points, normals, colors, pts_num=10000):
  points = points.reshape((pts_num, 3))
  normals = normals.reshape((pts_num, 3))
  colors = colors.reshape((pts_num, 3))
  data = np.concatenate([points, normals, colors], axis=1)
  assert data.shape[0] == pts_num

  header = "ply\n" \
           "format ascii 1.0\n" \
           "element vertex %d\n" \
           "property float x\n" \
           "property float y\n" \
           "property float z\n" \
           "property float nx\n" \
           "property float ny\n" \
           "property float nz\n" \
           "property float r\n" \
           "property float g\n" \
           "property float b\n" \
           "element face 0\n" \
           "property list uchar int vertex_indices\n" \
           "end_header\n"
  with open(filename, 'w') as fid:
    fid.write(header % pts_num)
    np.savetxt(fid, data, fmt='%.6f')


# run the experiments
if __name__ == '__main__':
  FLAGS = parse_args()
  compute_graph = ComputeGraphSeg(FLAGS)
  solver = PartNetSolver(FLAGS, compute_graph)
  solver.run()
