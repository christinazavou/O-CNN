import os

import tensorflow as tf
import numpy as np

from config import parse_args, FLAGS
from partnet_labels import find_category, LEVEL3_LABELS, LEVEL3_COLORS, decimal_to_rgb, get_level3_category_labels
from tfsolver import TFSolver
from network_factory import seg_network
from dataset import DatasetFactory
from ocnn import loss_functions_seg, build_solver, get_seg_label, loss_functions_seg_debug_checks
from libs import points_property, octree_property, octree_decode_key

# Add config
from visualize import vis_confusion_matrix

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
def tf_IoU_per_shape(pred, label, class_num, mask=-1, debug=False):
  debug_checks = {}
  with tf.name_scope('IoU'):
    # Set mask to 0 to filter unlabeled points, whose label is 0
    debug_checks['label_mask'] = label > mask  # mask out label
    debug_checks['prediction_masked'] = tf.boolean_mask(pred, debug_checks['label_mask'])
    debug_checks['label_masked'] = tf.boolean_mask(label, debug_checks['label_mask'])
    debug_checks['prediction_masked_argmax'] = tf.argmax(debug_checks['prediction_masked'],
                                                         axis=1, output_type=tf.int32)

    intsc, union = [None] * class_num, [None] * class_num
    for k in range(class_num):
      pk = tf.equal(debug_checks['prediction_masked_argmax'], k)
      lk = tf.equal(debug_checks['label_masked'], k)
      debug_checks['prediction_{}'.format(k)] = pk
      debug_checks['label_{}'.format(k)] = lk
      intsc[k] = tf.reduce_sum(tf.cast(pk & lk, dtype=tf.float32))
      union[k] = tf.reduce_sum(tf.cast(pk | lk, dtype=tf.float32))
    debug_checks['intsc'] = intsc
    debug_checks['union'] = union
  if debug:
    return debug_checks
  return intsc, union


# define the graph
class ComputeGraphSeg:
  def __init__(self, flags):
    self.flags = flags

  def create_dataset(self, flags_data):
    return DatasetFactory(flags_data)(return_iter=True)

  def __call__(self, dataset='train', training=True, reuse=False, gpu_num=1):

    if gpu_num != 1:
      raise Exception('Since I made a dict there is no implementation for multi gpu support')

    debug_checks = {}
    tensors_dict = {}

    FLAGS = self.flags
    with tf.device('/cpu:0'):
      flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
      data_iter = self.create_dataset(flags_data)

    with tf.device('/gpu:0'):
      with tf.name_scope('device_0'):
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
        metrics_dict, dc = loss_functions_seg_debug_checks(
          logit, label, FLAGS.LOSS.num_class, FLAGS.LOSS.weight_decay, 'ocnn', mask=0)
        debug_checks.update(dc)
        tensors_dict.update(metrics_dict)
        tensors_dict['total_loss'] = metrics_dict['loss'] + metrics_dict['regularizer']

        if flags_data.batch_size == 1:
          num_class = FLAGS.LOSS.num_class
          intsc, union = tf_IoU_per_shape(logit, label, num_class, mask=0)
          iou = tf.constant(0.0)     # placeholder, calc its value later
          tensors_dict['iou'] = iou
          for i in range(num_class):
            tensors_dict['intsc_%d' % i] = intsc[i]
            tensors_dict['union_%d' % i] = union[i]

    return tensors_dict, debug_checks


# define the solver
class PartNetSolver(TFSolver):
  def __init__(self, flags, compute_graph,  build_solver=build_solver):
    super(PartNetSolver, self).__init__(flags.SOLVER, compute_graph, build_solver)
    self.num_class = flags.LOSS.num_class # used to calculate the IoU

  def result_callback(self, avg_results_dict):
    # calc part-IoU, update `iou`, this is in correspondence with Line 77
    iou_avg = 0.0
    ious = [0] * self.num_class
    for i in range(1, self.num_class):  # !!! Ignore the first label
      instc_i = avg_results_dict['intsc_%d' % i]
      union_i = avg_results_dict['union_%d' % i]
      ious[i] = instc_i / (union_i + 1.0e-10)
      iou_avg = iou_avg + ious[i]
    iou_avg = iou_avg / (self.num_class - 1)
    avg_results_dict['iou'] = iou_avg
    return avg_results_dict

  def test(self):
    # build graph
    self.build_test_graph()

    # checkpoint
    assert(self.flags.ckpt)   # the self.flags.ckpt should be provided
    tf_saver = tf.train.Saver(max_to_keep=10)

    # start
    avg_test_dict = {key: np.zeros(value.get_shape()) for key, value in self.test_tensors_dict.items()}
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      test_keys = list(self.test_tensors_dict.keys())
      self.summ2txt(test_keys, 'batch')

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
        iter_test_result_dict, iter_tdc = sess.run([self.test_tensors_dict, self.test_debug_checks])

        points, labels, predictions, normals = iter_tdc['test/input_point_info/points'], \
                                               iter_tdc['test/input_point_info/labels'], \
                                               iter_tdc['softmax_loss/prediction'], \
                                               iter_tdc["test/input_point_info/normals"]

        dec_colors = LEVEL3_COLORS[category]
        cp = np.array([decimal_to_rgb(dec_colors[p]) for p in predictions])

        # note: since we have used a mask of 0, in points_label we ignore all points with 0 label i.e. that are undefined
        # so len(iter_debug_checks['softmax_loss/prediction']) != len(iter_debug_checks['test/input_point_info/points'])
        # so predictions and labels of those patches are ignored ..
        # but the metrics for all defined classes can still be calculated for this sample
        if predictions.shape[0] != points.shape[0]:
          label_mask = labels > 0
          normals, points = sess.run([tf.boolean_mask(normals, label_mask), tf.boolean_mask(points, label_mask)])
        points = points[:, 0: 3]

        assert cp.shape[0] == points.shape[0] == normals.shape[0]
        assert cp.shape[1] == points.shape[1] == normals.shape[1] == 3
        save_ply(os.path.join(predicted_ply_dir, "{}.ply".format(i)), points[:, 0:3], normals, cp)

        # run testing average and print the results
        reports = 'batch: %04d; ' % i
        for key, value in iter_test_result_dict.items():
          avg_test_dict[key] += value
          if key != 'confusion_matrix':
            reports += '%s: %0.4f; ' % (key, value)
        print(reports)

        # make sure results are sorted before writing them
        iter_test_result_sorted = []
        for key in test_keys:
          iter_test_result_sorted.append(iter_test_result_dict[key])
        self.summ2txt(iter_test_result_sorted, i)

    # Final testing results
    for key, value in avg_test_dict.items():
      avg_test_dict[key] /= self.flags.test_iter
    avg_test_dict = self.result_callback(avg_test_dict)

    # print the results
    print('Testing done!\n')
    reports = 'ALL: %04d; ' % self.flags.test_iter
    avg_test_sorted = []
    for key in test_keys:
      avg_test_sorted.append(avg_test_dict[key])
      if key != 'confusion_matrix':
        reports += '%s: %0.4f; ' % (key, avg_test_dict[key])
      else:
        vis_confusion_matrix(avg_test_dict[key].reshape(self.num_class, self.num_class),
                             get_level3_category_labels(category),
                             LEVEL3_COLORS[category])
    print(reports)
    self.summ2txt(avg_test_sorted, 'ALL')


def save_ply(filename, points, normals, colors):
  pts_num = points.shape[0]

  header = "ply\n" \
           "format ascii 1.0\n" \
           "element vertex %d\n" \
           "property float x\n" \
           "property float y\n" \
           "property float z\n" \
           "property float nx\n" \
           "property float ny\n" \
           "property float nz\n" \
           "property uchar r\n" \
           "property uchar g\n" \
           "property uchar b\n" \
           "element face 0\n" \
           "property list uchar int vertex_indices\n" \
           "end_header\n"
  with open(filename, 'w') as fid:
    fid.write(header % pts_num)
    for point, normal, color in zip(points, normals, colors):
      fid.write(" ".join([str(i) for i in point]) + " " +
                " ".join([str(i) for i in normal]) + " " +
                " ".join([str(int(i)) for i in color])+"\n")

# run the experiments
if __name__ == '__main__':
  FLAGS = parse_args()
  compute_graph = ComputeGraphSeg(FLAGS)
  solver = PartNetSolver(FLAGS, compute_graph)
  solver.run()
