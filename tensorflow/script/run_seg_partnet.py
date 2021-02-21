import time
from config import parse_args, parse_class_weights
from seg_helper import *
from tfsolver import TFSolver, DELIMITER
from network_factory import seg_network
from dataset_iterator import *
from dataset_preloader import *
from data_augmentation import *
from libs import points_property, octree_property
import numpy as np
from tqdm import trange
import os
from ocnn import *
from learning_rate import LRFactory
from tensorflow.python.client import timeline

import psutil
from memory_profiler import profile

# tf.compat.v1.enable_eager_execution()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Add config

best_metric_dict = {"accu": 0.0, "total_loss": 10e100, "iou": 0.0}  # TODO store last values in ckpt & restore them
DO_NOT_AVG = ["intsc_", "union_", "iou"]
TRAIN_DATA = DataLoader()
TEST_DATA = DataLoader()


# @profile
def read_datasets(flags):
    global TRAIN_DATA, TEST_DATA
    if flags.SOLVER.run != 'test':
        TRAIN_DATA = TRAIN_DATA(flags.DATA.train, flags.MODEL.nout, flags.MODEL.channel, True)
    if flags.SOLVER.run != "timeline":
        TEST_DATA = TEST_DATA(flags.DATA.test, flags.MODEL.nout, flags.MODEL.channel, True,
                              True if flags.SOLVER.run == "test" else False)


# get the label and pts
def get_point_info(points, mask_ratio=0, mask=-1):
    debug_checks = {}
    with tf.name_scope('points_info'):
        pts = points_property(points, property_name='xyz', channel=4)  # can run with channel=3 too, 4th channel is
        # point id (to which octree in the batch it belongs to)
        label = points_property(points, property_name='label', channel=1)
        debug_checks['{}/pts(xyz)'.format(tf.get_variable_scope().name)] = pts
        debug_checks['{}/label'.format(tf.get_variable_scope().name)] = label

        label = tf.reshape(label, [-1])
        label_mask = tf.not_equal(label, mask)  # mask out invalid points, -1

        if mask_ratio > 0:  # random drop some points to speed up training
            rnd_mask = tf.random.uniform(tf.shape(label_mask)) > mask_ratio
            label_mask = tf.logical_and(label_mask, rnd_mask)

        label = tf.boolean_mask(label, label_mask)
        pts = tf.boolean_mask(pts, label_mask)

        debug_checks['{}/masked_and_dropped/pts(xyz)'.format(tf.get_variable_scope().name)] = pts
        debug_checks['{}/masked_and_dropped/label'.format(tf.get_variable_scope().name)] = label
    return pts, label, debug_checks


def tf_IoU_per_shape(logits, label, class_num, mask=-1, ignore=666):
    # -1 CAN EXIST IF LABELS COME FROM OCTREE_PROPERTY('LABEL')
    with tf.name_scope('IoU'):
        # mask out unwanted labels (empty and undetermined)
        label_mask = tf.logical_and(tf.not_equal(label, mask), tf.not_equal(label, ignore))
        masked_label = tf.boolean_mask(label, label_mask)
        prediction = tf.argmax(tf.boolean_mask(logits, label_mask), axis=1, output_type=tf.int32)

        intsc, union = [0] * class_num, [0] * class_num
        for k in range(0, class_num):
            pk = tf.equal(prediction, k)
            lk = tf.equal(masked_label, k)
            intsc[k] = tf.reduce_sum(tf.cast(pk & lk, dtype=tf.float32))
            union[k] = tf.reduce_sum(tf.cast(pk | lk, dtype=tf.float32))
    return intsc, union


# define the graph
class ComputeGraphSeg:
    def __init__(self, flags):
        self.flags = flags
        self.weights = ComputeGraphSeg.set_weights(parse_class_weights(flags))

        self.points = tf.placeholder(dtype=tf.float32, name="batch_points")
        self.normals = tf.placeholder(dtype=tf.float32, name="point_nrms")
        self.features = tf.placeholder(dtype=tf.float32, name="point_fts")
        self.labels = tf.placeholder(dtype=tf.float32, name="point_labels")
        self.rot = tf.placeholder(dtype=tf.float32, name="point_rotation")

    @staticmethod
    def set_weights(w_list):
        weights = tf.constant(w_list)
        return weights

    def __call__(self, dataset='train', training=True, reuse=False, gpu_num=1):
        if gpu_num != 1:
            raise Exception('Since I made a dict there is no implementation for multi gpu support')

        debug_checks = {}
        tensors_dict = {}

        FLAGS = self.flags
        with tf.device('/cpu:0'):
            flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
            data_aug = DataAugmentor(flags_data)
        with tf.device('/gpu:0'):
            with tf.name_scope('device_0'):
                octree, points = data_aug(self.points, self.normals, self.features, self.labels, self.rot)

                print("mask ratio for {} is {}".format(dataset, flags_data.mask_ratio))
                pts, label, dc = get_point_info(points, flags_data.mask_ratio)
                debug_checks.update(dc)

                if not FLAGS.LOSS.point_wise:  # octree-wise loss
                    pts, label = None, get_seg_label(octree, FLAGS.MODEL.depth_out)
                    debug_checks["{}/seg_label/label(pts=None)".format(tf.get_variable_scope().name)] = label

                logit, dc = seg_network(octree, FLAGS.MODEL, training, reuse, pts=pts)
                debug_checks.update(dc)

                debug_checks["{}/probabilities".format(tf.get_variable_scope().name)] = get_probabilities(logit)

                metrics_dict = loss_functions_seg(logit=logit, label_gt=label, num_class=FLAGS.LOSS.num_class,
                                                  weight_decay=FLAGS.LOSS.weight_decay, var_name='ocnn',
                                                  weights=self.weights, mask=-1, ignore=FLAGS.MODEL.nout)
                tensors_dict.update(metrics_dict)
                tensors_dict['total_loss'] = metrics_dict['loss'] + metrics_dict['regularizer']

                if flags_data.batch_size == 1:  # TODO make it work for different batch sizes
                    num_class = FLAGS.LOSS.num_class
                    intsc, union = tf_IoU_per_shape(logit, label, num_class, mask=-1, ignore=FLAGS.MODEL.nout)
                    tensors_dict['iou'] = tf.constant(0.0)  # placeholder, calc its value later
                    for i in range(0, num_class):
                        tensors_dict['intsc_%d' % i] = intsc[i]
                        tensors_dict['union_%d' % i] = union[i]

        return tensors_dict, debug_checks


def result_callback(avg_results_dict, num_class):
    # calc part-IoU, update `iou`, this is in correspondence with Line 77
    ious = {}
    for i in range(0, num_class):  # !!! First label is wall, undetermined is num_class+1
        instc_i = avg_results_dict['intsc_%d' % i]
        union_i = avg_results_dict['union_%d' % i]  # max value of union is the # of determined points
        if union_i > 0.0:
            ious[i] = instc_i / union_i
    avg_results_dict['iou'] = sum(ious.values()) / len(ious)
    return avg_results_dict


def get_probabilities(logits):
    return tf.nn.softmax(logits)


# define the solver
class PartNetSolver(TFSolver):
    def __init__(self, flags, compute_graph, build_solver):
        self.flags = flags
        super(PartNetSolver, self).__init__(flags.SOLVER, compute_graph, build_solver)
        self.num_class = flags.LOSS.num_class  # used to calculate the IoU

    def result_callback(self, avg_results_dict):
        return result_callback(avg_results_dict, self.num_class)

    def build_train_graph(self):
        gpu_num = len(self.flags.gpu)
        train_params = {'dataset': 'train', 'training': True, 'reuse': False}
        test_params = {'dataset': 'test', 'training': False, 'reuse': True}
        if gpu_num > 1:  # TODO: check / remove / clean
            train_params['gpu_num'] = gpu_num
            test_params['gpu_num'] = gpu_num

        self.train_tensors_dict, self.train_debug_checks = self.graph(**train_params)
        self.test_tensors_dict, self.test_debug_checks = self.graph(**test_params)

        self.total_loss = self.train_tensors_dict['total_loss']
        with tf.name_scope('lr'):
            self.lr = tf.Variable(initial_value=self.flags.learning_rate, name='learning_rate', trainable=False)
        solver_param = [self.total_loss, self.lr]
        if gpu_num > 1:
            solver_param.append(gpu_num)
        self.train_op = self.build_solver(*solver_param)

        if gpu_num > 1:  # average the tensors from different gpus for summaries
            with tf.device('/cpu:0'):
                self.train_tensors_dict = average_tensors(self.train_tensors_dict)
                self.test_tensors_dict = average_tensors(self.test_tensors_dict)

        tensor_dict_for_test_summary = {}
        tensor_dict_for_test_summary.update(self.test_tensors_dict)
        tensor_dict_for_test_summary.update({'lr': self.lr})
        self.summaries_dict(self.train_tensors_dict, tensor_dict_for_test_summary)

    def summaries_dict(self, train_tensor_dict, test_tensor_dict):
        self.summ_train = summary_train_dict(train_tensor_dict)
        self.summ_test, self.summ_holder_dict = summary_test_dict(test_tensor_dict)
        self.csv_summ_test_keys = [key for key in self.summ_holder_dict.keys()]
        self.summ2txt(self.csv_summ_test_keys, 'iter', 'w')

    def build_test_graph(self):
        gpu_num = len(self.flags.gpu)
        test_params = {'dataset': 'test', 'training': False, 'reuse': False}
        if gpu_num > 1: test_params['gpu_num'] = gpu_num
        self.verbose = self.flags.verbose
        self.test_tensors_dict, self.test_debug_checks = self.graph(**test_params)
        if gpu_num > 1:  # average the tensors from different gpus
            with tf.device('/cpu:0'):
                self.test_tensors_dict = average_tensors(self.test_tensors_dict)

    # @profile
    def run_k_test_iterations(self, sess, test_batch):
        print("Running validation...")
        global TEST_DATA
        avg_results_dict = {key: np.zeros(value.get_shape()) for key, value in self.test_tensors_dict.items()}
        for _ in range(self.flags.test_iter):
            idxs, rots = sess.run(test_batch)
            pts, nrms, fts, labels = TEST_DATA.points[idxs], \
                                     TEST_DATA.normals[idxs], \
                                     TEST_DATA.features[idxs], \
                                     TEST_DATA.point_labels[idxs]

            iter_results_dict = sess.run(self.test_tensors_dict,
                                         feed_dict={self.graph.points: pts,
                                                    self.graph.normals: nrms,
                                                    self.graph.features: fts,
                                                    self.graph.labels: labels,
                                                    self.graph.rot: rots
                                                    })

            for key, value in iter_results_dict.items():
                avg_results_dict[key] += value

        for key in avg_results_dict.keys():
            if not any(k in key for k in DO_NOT_AVG):
                avg_results_dict[key] /= self.flags.test_iter
        # calculate iou
        avg_results = self.result_callback(avg_results_dict)

        curr_lr = sess.run(self.lr, feed_dict={self.lr: self.lr_metric(avg_results_dict['total_loss'])})
        sess.run(self.lr.assign(curr_lr))
        avg_results['lr'] = curr_lr
        return avg_results

    def save_ckpt(self, dc, sess, iter):
        with open(os.path.join(self.best_ckpt_path, "evaluation_report.txt"), 'a') as f:
            f.write('---- EPOCH %04d EVALUATION ----\n' % (iter))

            for key in best_metric_dict.keys():
                if not "loss" in key:
                    if dc[key] > best_metric_dict[key]:
                        best_metric_dict[key] = dc[key]
                        self.tf_saver.save(sess, save_path=os.path.join(self.best_ckpt_path, 'best_' + key + '.ckpt'),
                                           write_meta_graph=False)
                else:
                    if dc[key] < best_metric_dict[key]:
                        best_metric_dict[key] = dc[key]
                        self.tf_saver.save(sess, save_path=os.path.join(self.best_ckpt_path, 'best_' + key + '.ckpt'),
                                           write_meta_graph=False)

                f.write('eval ' + key + ': %f\n' % (dc[key]))

    # @profile
    def train(self):
        global TRAIN_DATA, TEST_DATA
        train_iter = DataIterator(TRAIN_DATA.flags, TRAIN_DATA.tfrecord_num)
        test_iter = DataIterator(TEST_DATA.flags, TEST_DATA.tfrecord_num)

        # build the computation graph
        self.build_train_graph()

        # checkpoint
        start_iter = 1
        self.tf_saver = tf.train.Saver(max_to_keep=self.flags.ckpt_num)

        ckpt_path = os.path.join(self.flags.logdir, 'model')
        self.best_ckpt_path = os.path.join(self.flags.logdir, 'best_ckpts')
        os.makedirs(self.best_ckpt_path, exist_ok=True)
        if self.flags.ckpt:  # restore from the provided checkpoint
            ckpt = self.flags.ckpt
        else:  # restore from the breaking pointer
            ckpt = tf.train.latest_checkpoint(ckpt_path)
            if ckpt: start_iter = int(ckpt[ckpt.find("iter") + 5:-5]) + 1

        # session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            summary_writer = tf.summary.FileWriter(self.flags.logdir, sess.graph)

            print('Initialize ...')
            self.initialize(sess)

            if ckpt:
                self.restore(sess, ckpt)
            py = psutil.Process(os.getpid())
            memory_usage = py.memory_info()[0] / 1024 ** 3
            print("init memory: ", memory_usage)
            print('Start training ...')
            # option 1: use feed dict to pass the calculated learning rate
            # option 2: use model.compile and pass the optimizer and the callbacks
            if ckpt:
                self.flags.defrost()
                print(self.flags.learning_rate)
                self.flags.learning_rate = float(sess.run(self.lr))
                print(self.flags.learning_rate)
                self.flags.freeze()

            self.lr_metric = LRFactory(self.flags)
            batch = train_iter()
            test_batch = test_iter()

            for i in trange(start_iter, self.flags.max_iter + 1, ncols=80, desc="Train"):
                idxs, rots = sess.run(batch)
                pts, nrms, fts, labels = TRAIN_DATA.points[idxs], \
                                         TRAIN_DATA.normals[idxs], \
                                         TRAIN_DATA.features[idxs], \
                                         TRAIN_DATA.point_labels[idxs]
                # training
                summary_train, _, curr_loss, curr_lr = sess.run(
                    [self.summ_train, self.train_op, self.total_loss, self.lr],
                    feed_dict={self.graph.points: pts,
                               self.graph.normals: nrms,
                               self.graph.features: fts,
                               self.graph.labels: labels,
                               self.graph.rot: rots
                               })
                py = psutil.Process(os.getpid())
                memory_usage = py.memory_info()[0] / 1024 ** 3
                print("iter memory: ", memory_usage)

                summary_writer.add_summary(summary_train, i)

                # testing
                if i % self.flags.test_every_iter == 0:
                    # run testing average
                    avg_test_dict = self.run_k_test_iterations(sess, test_batch)

                    # save best acc,loss and iou network snapshots
                    self.save_ckpt(avg_test_dict, sess, i / self.flags.test_every_iter)

                    # run testing summary
                    summary = sess.run(self.summ_test,
                                       feed_dict={pl: avg_test_dict[m]
                                                  for m, pl in self.summ_holder_dict.items()})
                    summary_writer.add_summary(summary, i)
                    csv_avg_tests = [avg_test_dict[key] for key in self.csv_summ_test_keys]
                    self.summ2txt(csv_avg_tests, i)

                    # save session
                    ckpt_name = os.path.join(ckpt_path, 'iter_%06d.ckpt' % i)
                    self.tf_saver.save(sess, ckpt_name, write_meta_graph=False)

            print('Training done!')

    def timeline(self):
        global TRAIN_DATA
        train_iter = DataIterator(TRAIN_DATA.flags, TRAIN_DATA.tfrecord_num)
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
            batch = train_iter()
            print('Start profiling ...')
            for i in trange(0, timeline_skip + timeline_iter, ncols=80, desc="Timeline"):
                idxs, rots = sess.run(batch)
                pts, nrms, fts, labels = TRAIN_DATA.points[idxs], \
                                         TRAIN_DATA.normals[idxs], \
                                         TRAIN_DATA.features[idxs], \
                                         TRAIN_DATA.point_labels[idxs]

                summary, _ = sess.run([self.summ_train, self.train_op],
                                      options=options, run_metadata=run_metadata,
                                      feed_dict={self.graph.points: pts,
                                                 self.graph.normals: nrms,
                                                 self.graph.features: fts,
                                                 self.graph.labels: labels,
                                                 self.graph.rot: rots
                                                 })
                if i == timeline_skip + timeline_iter - 1:
                    # summary_writer.add_run_metadata(run_metadata, 'step_%d'%i, i)
                    # write timeline to a json file
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(os.path.join(self.flags.logdir, 'timeline.json'), 'w') as f:
                        f.write(chrome_trace)
                summary_writer.add_summary(summary, i)
            print('Profiling done!')

    def test(self):
        # CATEGORIES = ANNFASS_LABELS
        COLOURS = ANNFASS_COLORS
        global TEST_DATA
        test_iter = DataIterator(TEST_DATA.flags, TEST_DATA.tfrecord_num)
        # build graph
        self.build_test_graph()

        # checkpoint
        assert self.flags.ckpt, "Checkpoint was not provided!!!"  # the self.flags.ckpt should be provided
        assert self.flags.test_iter == len(
            TEST_DATA.filenames), "Test iterations does not match number of files provided ({} vs {})".format(
            self.flags.test_iter, len(TEST_DATA.filenames))
        tf_saver = tf.train.Saver(max_to_keep=10)
        logdir = os.path.join(self.flags.logdir, os.path.basename(self.flags.ckpt).split(".")[0])

        # start
        test_metrics_dict = {key: np.zeros(value.get_shape()) for key, value in self.test_tensors_dict.items()}
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.summ2txt_line(self.flags.ckpt)
            self.summ2txt_line(DELIMITER.join(['iteration'] + [key for key in self.test_tensors_dict.keys()]))

            # restore and initialize
            self.initialize(sess)
            print('Restore from checkpoint: %s' % self.flags.ckpt)
            tf_saver.restore(sess, self.flags.ckpt)

            predicted_ply_dir = os.path.join(logdir, "predicted_ply")
            if not os.path.exists(predicted_ply_dir):
                os.makedirs(predicted_ply_dir)
            probabilities_dir = os.path.join(logdir, "probabilities")
            if not os.path.exists(probabilities_dir):
                os.makedirs(probabilities_dir)

            batch = test_iter()
            print('Start testing ...')
            for i in range(0, self.flags.test_iter):
                idxs, rots = sess.run(batch)
                pts, nrms, fts, labels = TEST_DATA.points[idxs], \
                                         TEST_DATA.normals[idxs], \
                                         TEST_DATA.features[idxs], \
                                         TEST_DATA.point_labels[idxs]

                iter_test_result_dict, iter_tdc = sess.run([self.test_tensors_dict, self.test_debug_checks],
                                                           feed_dict={self.graph.points: pts,
                                                                      self.graph.normals: nrms,
                                                                      self.graph.features: fts,
                                                                      self.graph.labels: labels,
                                                                      self.graph.rot: rots
                                                                      })
                iter_test_result_dict = self.result_callback(iter_test_result_dict)

                probabilities = iter_tdc['/probabilities']
                filename = os.path.basename(TEST_DATA.filenames[i])
                predictions = np.argmax(probabilities, axis=1).astype(np.int32) + 1  # remap labels to initial values
                prediction_colors = np.array([to_rgb(COLOURS[int(p)]) for p in predictions])

                if self.verbose:
                    reports = str(i) + "-" + filename + ": "
                    for key, value in iter_test_result_dict.items():
                        test_metrics_dict[key] += value
                        reports += '%s: %0.4f; ' % (key, value)
                    print(reports)
                else:
                    for key, value in iter_test_result_dict.items():
                        test_metrics_dict[key] += value

                current_ply_o_f = os.path.join(predicted_ply_dir, "{}.ply".format(filename))
                save_ply(current_ply_o_f, pts[0], prediction_colors)

                np.save(file=os.path.join(probabilities_dir, filename), arr=np.array(probabilities))
                self.summ2txt([value for key, value in iter_test_result_dict.items()], str(i) + "-" + filename)

        # Average test results
        for key, value in test_metrics_dict.items():
            test_metrics_dict[key] /= self.flags.test_iter
        test_metrics_dict = self.result_callback(test_metrics_dict)

        # print the results
        avg_test_sorted = []
        print('Testing done!\n')
        if self.verbose:
            reports = 'ALL: %04d; ' % self.flags.test_iter
            for key in iter_test_result_dict.keys():
                avg_test_sorted.append(test_metrics_dict[key])
                reports += '%s: %0.4f; ' % (key, test_metrics_dict[key])
            print(reports)
        else:
            for key in iter_test_result_dict.keys():
                avg_test_sorted.append(test_metrics_dict[key])
        self.summ2txt(avg_test_sorted, 'ALL')


# run the experiments
if __name__ == '__main__':
    t = time.time()
    FLAGS = parse_args()
    print("\nReading data files. This may take a while...\n")
    read_datasets(FLAGS)
    compute_graph = ComputeGraphSeg(FLAGS)
    builder_op = build_solver_given_lr if FLAGS.SOLVER.lr_type == 'plateau' else build_solver
    solver = PartNetSolver(FLAGS, compute_graph, builder_op)
    solver.run()
    print("Minutes passed {}".format((time.time() - t) / 60))
