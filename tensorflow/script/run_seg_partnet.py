import os

import tensorflow as tf
import numpy as np
import pickle

from config import parse_args, FLAGS
from seg_labels import find_category, LEVEL3_LABELS, LEVEL3_COLORS, decimal_to_rgb, get_level3_category_labels, \
  ANNFASS_COLORS, ANNFASS_LABELS, to_rgb
from tfsolver import TFSolver
from network_factory import seg_network
from dataset import DatasetFactory
from libs import points_property, octree_property, octree_decode_key
import numpy as np
from tqdm import tqdm
import os
from ocnn import *
from learning_rate import LRFactory
from tensorflow.python.client import timeline

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Add config
from visualize import vis_confusion_matrix

# FLAGS.LOSS.point_wise = True
FLAGS.LOSS.point_wise = False
MASK_LABEL = 0  # metrics are ignored for the points with label 'undefined' ..
CONF_MAT_KEY = 'confusion_matrix'
# CATEGORIES = ANNFASS_LABELS
CATEGORIES = LEVEL3_LABELS
# COLOURS = ANNFASS_COLORS
COLOURS = LEVEL3_COLORS


# get the label and pts
def get_point_info(points, mask_ratio=0, mask=-1):
  debug_checks = {}
  with tf.name_scope('points_info'):
    pts   = points_property(points, property_name='xyz', channel=4)
    label = points_property(points, property_name='label', channel=1)
    debug_checks['{}/pts(xyz)'.format(tf.get_variable_scope().name)] = pts
    debug_checks['{}/label'.format(tf.get_variable_scope().name)] = label
    label = tf.reshape(label, [-1])
    label_mask = label > mask  # mask out invalid points, -1
    if mask_ratio > 0:         # random drop some points to speed up training
      rnd_mask = tf.random.uniform(tf.shape(label_mask)) > mask_ratio
      label_mask = tf.logical_and(label_mask, rnd_mask)
    pts   = tf.boolean_mask(pts, label_mask)
    label = tf.boolean_mask(label, label_mask)
    debug_checks['{}/masked_and_ratio/pts(xyz)'.format(tf.get_variable_scope().name)] = pts
    debug_checks['{}/masked_and_ratio/label'.format(tf.get_variable_scope().name)] = label
  return pts, label, debug_checks


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
        debug_checks["{}/octree".format(tf.get_variable_scope().name)] = octree
        debug_checks["{}/points".format(tf.get_variable_scope().name)] = points
        debug_checks["{}/labels".format(tf.get_variable_scope().name)] = _labels

        print("mask ratio for {} is {}".format(dataset, flags_data.mask_ratio))
        pts, label, dc = get_point_info(points, flags_data.mask_ratio)
        debug_checks.update(dc)
        debug_checks["{}/normals".format(tf.get_variable_scope().name)] = points_property(
          points, property_name='normal', channel=3)

        if not FLAGS.LOSS.point_wise:
          pts, label = None, get_seg_label(octree, FLAGS.MODEL.depth_out)
          debug_checks["{}/seg_label/label(pts=None)".format(tf.get_variable_scope().name)] = label

        logit, dc = seg_network(octree, FLAGS.MODEL, training, reuse, pts=pts)
        debug_checks.update(dc)
        debug_checks["{}/logit".format(tf.get_variable_scope().name)] = logit
        metrics_dict, dc = loss_functions_seg_debug_checks(
          logit, label, FLAGS.LOSS.num_class, FLAGS.LOSS.weight_decay, 'ocnn', mask=MASK_LABEL)
        debug_checks.update(dc)
        tensors_dict.update(metrics_dict)
        tensors_dict['total_loss'] = metrics_dict['loss'] + metrics_dict['regularizer']

        if flags_data.batch_size == 1:
          num_class = FLAGS.LOSS.num_class
          intsc, union = tf_IoU_per_shape(logit, label, num_class, mask=MASK_LABEL)
          iou = tf.constant(0.0)     # placeholder, calc its value later
          tensors_dict['iou'] = iou
          for i in range(num_class):
            tensors_dict['intsc_%d' % i] = intsc[i]
            tensors_dict['union_%d' % i] = union[i]

    return tensors_dict, debug_checks


def result_callback(avg_results_dict, num_class):
  try:
    return result_callback_maria(avg_results_dict, num_class)
  except Exception as e:
    raise Exception("Got exception: {}. Maybe you didnt use 'DATA.test.batch_size 1'".format(e))
    # # calc part-IoU, update `iou`, this is in correspondence with Line 77
    # iou_avg = 0.0
    # ious = [0] * num_class
    # for i in range(1, num_class):  # !!! Ignore the first label
    #     instc_i = avg_results_dict['intsc_%d' % i]
    #     union_i = avg_results_dict['union_%d' % i]
    #     ious[i] = instc_i / (union_i + 1.0e-10)
    #     iou_avg = iou_avg + ious[i]
    # iou_avg = iou_avg / (num_class - 1)
    # avg_results_dict['iou'] = iou_avg
    # return avg_results_dict


def result_callback_maria(avg_results_dict, num_class):
  # calc part-IoU, update `iou`, this is in correspondence with Line 77
  iou_avg = 0.0
  ious = [0] * num_class
  for i in range(1, num_class):  # !!! Ignore the first label
    instc_i = avg_results_dict['intsc_%d' % i]
    union_i = avg_results_dict['union_%d' % i]
    if union_i > 0.0:
      ious[i] = instc_i / union_i
    else:
      ious[i] = 0.0
    iou_avg = iou_avg + ious[i]
  iou_avg = iou_avg / np.count_nonzero(ious)
  avg_results_dict['iou'] = iou_avg
  return avg_results_dict


# define the solver
class PartNetSolver(TFSolver):
  def __init__(self, flags, compute_graph,  build_solver=build_solver):
    super(PartNetSolver, self).__init__(flags.SOLVER, compute_graph, build_solver)
    self.num_class = flags.LOSS.num_class # used to calculate the IoU

  def result_callback(self, avg_results_dict):
    return result_callback(avg_results_dict, self.num_class)

  def build_train_graph(self):
    gpu_num = len(self.flags.gpu)
    train_params = {'dataset': 'train', 'training': True,  'reuse': False}
    test_params  = {'dataset': 'test',  'training': False, 'reuse': True}
    if gpu_num > 1:
      train_params['gpu_num'] = gpu_num
      test_params['gpu_num']  = gpu_num

    self.train_tensors_dict, self.train_debug_checks = self.graph(**train_params)
    self.test_tensors_dict, self.test_debug_checks = self.graph(**test_params)

    total_loss = self.train_tensors_dict['total_loss']
    solver_param = [total_loss, LRFactory(self.flags)]
    if gpu_num > 1:
      solver_param.append(gpu_num)
    self.train_op, lr = self.build_solver(*solver_param)

    if gpu_num > 1: # average the tensors from different gpus for summaries
      with tf.device('/cpu:0'):
        self.train_tensors_dict = average_tensors(self.train_tensors_dict)
        self.test_tensors_dict = average_tensors(self.test_tensors_dict)

    tensor_dict_for_train_summary = {}
    tensor_dict_for_train_summary.update(self.train_tensors_dict)
    tensor_dict_for_train_summary.update({'lr': lr})
    tensor_dict_for_test_summary = {}
    tensor_dict_for_test_summary.update(self.test_tensors_dict)
    self.summaries_dict(tensor_dict_for_train_summary, tensor_dict_for_test_summary)

  def summaries_dict(self, train_tensor_dict, test_tensor_dict):
    self.summ_train_occ = None
    if CONF_MAT_KEY in train_tensor_dict:
      self.summ_train_occ = summary_train_dict({CONF_MAT_KEY: train_tensor_dict[CONF_MAT_KEY]})
      del train_tensor_dict[CONF_MAT_KEY]
    self.summ_train_alw = summary_train_dict(train_tensor_dict)
    self.summ_test, self.summ_holder_dict = summary_test_dict(test_tensor_dict)
    self.summ_test_keys = [key for key in self.summ_holder_dict.keys() if key != CONF_MAT_KEY]
    self.summ2txt(self.summ_test_keys, 'step', 'w')

  def build_test_graph(self):
    gpu_num = len(self.flags.gpu)
    test_params  = {'dataset': 'test',  'training': False, 'reuse': False}
    if gpu_num > 1: test_params['gpu_num'] = gpu_num
    self.test_tensors_dict, self.test_debug_checks = self.graph(**test_params)
    if gpu_num > 1: # average the tensors from different gpus
      with tf.device('/cpu:0'):
        self.test_tensors_dict = average_tensors(self.test_tensors_dict)

  def run_k_test_iterations(self, sess):
    avg_results_dict = {key: np.zeros(value.get_shape()) for key, value in self.test_tensors_dict.items()}

    for i in range(self.flags.test_iter):
      iter_results_dict, iter_debug_checks = sess.run([self.test_tensors_dict, self.test_debug_checks])

      for key, value in iter_results_dict.items():
        avg_results_dict[key] += value

    for key in avg_results_dict.keys():
      avg_results_dict[key] /= self.flags.test_iter
    avg_results = self.result_callback(avg_results_dict)
    return avg_results

  def train(self):
    # build the computation graph
    self.build_train_graph()

    # checkpoint
    start_iter = 1
    self.tf_saver = tf.train.Saver(max_to_keep=self.flags.ckpt_num)
    ckpt_path = os.path.join(self.flags.logdir, 'model')
    if self.flags.ckpt:  # restore from the provided checkpoint
      ckpt = self.flags.ckpt
    else:  # restore from the breaking point
      ckpt = tf.train.latest_checkpoint(ckpt_path)
      if ckpt: start_iter = int(ckpt[ckpt.find("iter") + 5:-5]) + 1

    # session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)

      print('Initialize ...')
      self.initialize(sess)
      if ckpt: self.restore(sess, ckpt)

      print('Start training ...')
      for i in tqdm(range(start_iter, self.flags.max_iter + 1), ncols=80):
        # training
        if self.summ_train_occ != None:
          summary_alw, summary_occ, _ = sess.run([self.summ_train_alw, self.summ_train_occ, self.train_op])
          summary_writer.add_summary(summary_occ, i)
          summary_writer.add_summary(summary_alw, i)
        else:
          summary_alw, _ = sess.run([self.summ_train_alw, self.train_op])
          summary_writer.add_summary(summary_alw, i)

        # testing
        if i % self.flags.test_every_iter == 0:
          # run testing average
          avg_test_dict = self.run_k_test_iterations(sess)

          # run testing summary
          summary = sess.run(self.summ_test,
                             feed_dict={pl: avg_test_dict[m]
                                        for m, pl in self.summ_holder_dict.items()})
          summary_writer.add_summary(summary, i)
          avg_test_ordered = []
          for key in self.summ_test_keys:
            avg_test_ordered.append(avg_test_dict[key])
          self.summ2txt(avg_test_ordered, i)

          # save session
          ckpt_name = os.path.join(ckpt_path, 'iter_%06d.ckpt' % i)
          self.tf_saver.save(sess, ckpt_name, write_meta_graph=False)

      print('Training done!')

  def timeline(self):
    # build the computation graph
    self.build_train_graph()

    # session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    timeline_skip, timeline_iter = 100, 2
    with tf.Session(config=config) as sess:
      summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)
      print('Initialize ...')
      self.initialize(sess)

      print('Start profiling ...')
      for i in tqdm(range(0, timeline_skip + timeline_iter), ncols=80):
        if i < timeline_skip:
          summary_alw, _ = sess.run([self.summ_train_alw, self.train_op])
        else:
          summary, _ = sess.run([self.summ_train_alw, self.train_op],
                                options=options, run_metadata=run_metadata)
          if (i == timeline_skip + timeline_iter - 1):
            # summary_writer.add_run_metadata(run_metadata, 'step_%d'%i, i)
            # write timeline to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(os.path.join(self.flags.logdir, 'timeline.json'), 'w') as f:
              f.write(chrome_trace)
        summary_writer.add_summary(summary, i)
      print('Profiling done!')

  def test(self):
    # build graph
    self.build_test_graph()

    # checkpoint
    assert(self.flags.ckpt)   # the self.flags.ckpt should be provided
    tf_saver = tf.train.Saver(max_to_keep=10)

    # start
    test_metrics_dict = {key: np.zeros(value.get_shape()) for key, value in self.test_tensors_dict.items()}
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
      test_keys = list(self.test_tensors_dict.keys())
      self.summ2txt([key for key in test_keys if key != CONF_MAT_KEY], 'batch')

      # restore and initialize
      self.initialize(sess)
      print('Restore from checkpoint: %s' % self.flags.ckpt)
      tf_saver.restore(sess, self.flags.ckpt)

      category = find_category(self.flags.ckpt, CATEGORIES)
      assert category is not None
      predicted_ply_dir = os.path.join(self.flags.logdir, "predicted_ply_{}".format(category))
      if not os.path.exists(predicted_ply_dir):
        os.makedirs(predicted_ply_dir)
      groundtruth_ply_dir = os.path.join(self.flags.logdir, "groundtruth_ply_{}".format(category))
      if not os.path.exists(groundtruth_ply_dir):
        os.makedirs(groundtruth_ply_dir)
      predicted_pkl_dir = os.path.join(self.flags.logdir, "predicted_pkl_{}".format(category))
      if not os.path.exists(predicted_pkl_dir):
        os.makedirs(predicted_pkl_dir)
      probabilities_pkl_dir = os.path.join(self.flags.logdir, "probabilities_pkl_{}".format(category))
      if not os.path.exists(probabilities_pkl_dir):
        os.makedirs(probabilities_pkl_dir)

      print('Start testing ...')
      for i in range(0, self.flags.test_iter):
        iter_test_result_dict, iter_tdc = sess.run([self.test_tensors_dict, self.test_debug_checks])
        iter_test_result_dict = self.result_callback(iter_test_result_dict)

        points, labels, normals, logit, masked_logit = iter_tdc['/pts(xyz)'][:, 0:3], \
                                                       iter_tdc['/label'], \
                                                       iter_tdc["/normals"], \
                                                       iter_tdc["/logit"], \
                                                       iter_tdc['/masked_logit']
        predictions = np.argmax(logit, axis=1).astype(np.int32)
        probabilities = []
        for point in range(logit.shape[0]):
          probabilities.append(logit[point, predictions[point]])
        probabilities = np.array(probabilities)

        l_colors = np.array([to_rgb(COLOURS[category][int(l)]) if l >= 0 else to_rgb(COLOURS[category][0])
                             for l in labels])
        p_colors = np.array([to_rgb(COLOURS[category][int(p)]) if p >= 0 else to_rgb(COLOURS[category][0])
                             for p in predictions])

        masked_predictions = np.argmax(masked_logit, axis=1).astype(np.int32)
        # if predictions.shape != masked_predictions.shape:
        #   print("!=:", labels.min(), labels.max(), predictions.min(), predictions.max())

        # run testing average and print the results
        reports = 'batch: %04d; ' % i
        for key, value in iter_test_result_dict.items():
          test_metrics_dict[key] += value
          if key != CONF_MAT_KEY:
            reports += '%s: %0.4f; ' % (key, value)
        print(reports)

        current_iou = int(self.result_callback(iter_test_result_dict)['iou'] * 100)
        current_ply_i_f = os.path.join(groundtruth_ply_dir, "i{}.ply".format(i))
        current_ply_o_f = os.path.join(predicted_ply_dir, "iou{}_i{}.ply".format(current_iou, i))
        probabilities_o_f = os.path.join(probabilities_pkl_dir, "i{}.ply".format(i))
        save_ply(current_ply_i_f, points, normals, l_colors)
        save_ply(current_ply_o_f, points, normals, p_colors)
        current_pkl_f = os.path.join(predicted_pkl_dir, "p{}_m{}_i{}.pkl".format(predictions.shape[0],
                                                                                 masked_predictions.shape[0], i))
        save_pickled(current_pkl_f, labels.ravel(), predictions.ravel())
        save_pickled_np(probabilities_o_f, probabilities)

        # make sure results are sorted before writing them
        iter_test_result_sorted = []
        for key in test_keys:
          if key != CONF_MAT_KEY:
            iter_test_result_sorted.append(iter_test_result_dict[key])
        self.summ2txt(iter_test_result_sorted, i)

    # Final testing results
    for key, value in test_metrics_dict.items():
      if key != CONF_MAT_KEY:
        test_metrics_dict[key] /= self.flags.test_iter
    test_metrics_dict = self.result_callback(test_metrics_dict)

    # print the results
    print('Testing done!\n')
    reports = 'ALL: %04d; ' % self.flags.test_iter
    avg_test_sorted = []
    for key in test_keys:
      if key != CONF_MAT_KEY:
        avg_test_sorted.append(test_metrics_dict[key])
        reports += '%s: %0.4f; ' % (key, test_metrics_dict[key])
      else:
        vis_confusion_matrix(test_metrics_dict[key].reshape(self.num_class, self.num_class),
                             CATEGORIES[category],
                             COLOURS[category],
                             "Category: {}, Test samples: {}".format(category, self.flags.test_iter))
    print(reports)
    self.summ2txt(avg_test_sorted, 'ALL')


def save_pickled(filename, ground_truth, prediction):
  # assert len(ground_truth.shape) == 1 and len(prediction.shape) == 1
  with open(filename, 'wb') as fout:
    pickle.dump({'orig': list(ground_truth), 'pred': list(prediction)}, fout)

def save_pickled_np(filename, probabilities):
  pickle.dump(probabilities, open(filename, 'wb'))

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
