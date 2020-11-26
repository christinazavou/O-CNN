import pickle
import time
from config import parse_args, FLAGS, parse_class_weights
from seg_helper import decimal_to_rgb, ANNFASS_COLORS, ANNFASS_LABELS, to_rgb, save_ply
from tfsolver import TFSolver, DELIMITER
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

FLAGS.LOSS.point_wise = True
# FLAGS.LOSS.point_wise = False
IGNORE_LABELS = tf.constant([0.0, 32.0, 33.0])  # metrics are ignored for the points with label 'undefined' ..
CONF_MAT_KEY = 'confusion_matrix'
TEST_SPLIT = "./configs/test_split.txt"
CATEGORIES = ANNFASS_LABELS
COLOURS = ANNFASS_COLORS
best_metric_dict = {"acc": 0.0, "loss": 1e100, "iou": 0.0}
DO_NOT_AVG = ["intsc_", "union_", "iou"]


# get the label and pts
def get_point_info(points, mask_ratio=0, mask=-1):
    debug_checks = {}
    with tf.name_scope('points_info'):
        pts = points_property(points, property_name='xyz', channel=4)
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


def tf_IoU_per_shape(logits, label, class_num, mask=-1, ignore=0):
    # -1 CAN EXIST IF LABELS COME FROM OCTREE_PROPERTY('LABEL') ...
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
        self.weights=ComputeGraphSeg.set_weights(parse_class_weights(flags))

    @staticmethod
    def set_weights( w_list):
        weights = tf.constant(w_list)
        return weights

    def create_dataset(self, flags_data):
        return DatasetFactory(flags_data)(return_iter=True)

    def __call__(self, dataset='train', training=True, reuse=False, gpu_num=1):

        if gpu_num != 1:
            raise Exception('Since I made a dict there is no implementation for multi gpu support')

        debug_checks = {}
        tensors_dict = {}

        train_mode = True if dataset == 'train' else False

        FLAGS = self.flags
        # print(FLAGS)
        with tf.device('/cpu:0'):
            flags_data = FLAGS.DATA.train if dataset == 'train' else FLAGS.DATA.test
            data_iter = self.create_dataset(flags_data)

        with tf.device('/gpu:0'):
            with tf.name_scope('device_0'):
                octree, _, points = data_iter.get_next()

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
                                                  weights=self.weights, mask=-1, ignore=0, train_mode=train_mode)
                tensors_dict.update(metrics_dict)
                tensors_dict['total_loss'] = metrics_dict['loss'] + metrics_dict['regularizer']

                if flags_data.batch_size == 1:
                    num_class = FLAGS.LOSS.num_class
                    intsc, union = tf_IoU_per_shape(logit, label, num_class, mask=-1, ignore=0)
                    tensors_dict['iou'] = tf.constant(0.0)  # placeholder, calc its value later
                    for i in range(0, num_class):
                        tensors_dict['intsc_%d' % i] = intsc[i]
                        tensors_dict['union_%d' % i] = union[i]

        return tensors_dict, debug_checks


def result_callback(avg_results_dict, num_class):
    # calc part-IoU, update `iou`, this is in correspondence with Line 77
    ious = [0] * num_class
    for i in range(0, num_class):  # !!! Ignore the first label
        instc_i = avg_results_dict['intsc_%d' % i]
        union_i = avg_results_dict['union_%d' % i]
        if union_i > 0.0:
            ious[i] = instc_i / union_i
    avg_results_dict['iou'] = sum(ious) / (num_class - 1)
    return avg_results_dict


def get_probabilities(logits):
    return tf.nn.softmax(logits)


# define the solver
class PartNetSolver(TFSolver):
    def __init__(self, flags, compute_graph, build_solver):
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
        self.csv_summ_test_keys = [key for key in self.summ_holder_dict.keys() if key != CONF_MAT_KEY]
        self.summ2txt(self.csv_summ_test_keys, 'iter', 'w')

    def build_test_graph(self):
        gpu_num = len(self.flags.gpu)
        test_params = {'dataset': 'test', 'training': False, 'reuse': False}
        if gpu_num > 1: test_params['gpu_num'] = gpu_num
        self.test_tensors_dict, self.test_debug_checks = self.graph(**test_params)
        if gpu_num > 1:  # average the tensors from different gpus
            with tf.device('/cpu:0'):
                self.test_tensors_dict = average_tensors(self.test_tensors_dict)

    def run_k_test_iterations(self, sess):
        print("Evaluation (val split)")
        avg_results_dict = {key: np.zeros(value.get_shape()) for key, value in self.test_tensors_dict.items()}
        for i in range(self.flags.test_iter):
            # iter_results_dict, iter_debug_checks = sess.run([self.test_tensors_dict, self.test_debug_checks])
            iter_results_dict = sess.run(self.test_tensors_dict)

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
        if dc['iou'] > best_metric_dict['iou']:
            # print(best_metric_dict['iou'], dc['iou'])
            best_metric_dict['iou'] = dc['iou']
            self.tf_saver.save(sess, save_path=os.path.join(self.best_ckpt_path, 'best_iou.ckpt'))
        if dc['accu'] > best_metric_dict['acc']:
            # print(best_metric_dict['acc'], dc['accu'])
            best_metric_dict['acc'] = dc['accu']
            self.tf_saver.save(sess, save_path=os.path.join(self.best_ckpt_path, "best_acc.ckpt"))
        if dc['total_loss'] < best_metric_dict['loss']:
            # print(best_metric_dict['loss'], dc['total_loss'])
            best_metric_dict['loss'] = dc['total_loss']
            self.tf_saver.save(sess, save_path=os.path.join(self.best_ckpt_path, "best_loss.ckpt"))

        with open(os.path.join(self.best_ckpt_path, "evaluation_report.txt"), 'a') as f:
            f.write('---- EPOCH %04d EVALUATION ----\n' % (iter))
            f.write('eval accuracy: %f\n' % (dc['accu']))
            f.write('eval iou: %f\n' % (dc['iou']))
            f.write('eval loss: %f\n' % (dc['total_loss']))

    def train(self):
        # build the computation graph
        self.build_train_graph()

        # checkpoint
        start_iter = 1
        self.tf_saver = tf.train.Saver(max_to_keep=self.flags.ckpt_num)

        ckpt_path = os.path.join(self.flags.logdir, 'model')
        self.best_ckpt_path = os.path.join(self.flags.logdir, 'best_ckpts')
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

            print('Start training ...')
            # option 1: use feed dict to pass the calculated learning rate
            # option 2: use model.compile and pass the optimizer and the callbacks
            if ckpt:
                self.flags.defrost()
                print(self.flags.learning_rate)
                self.flags.learning_rate = float(sess.run(self.lr))
                print(self.flags.learning_rate)
                self.flags.freeze()

            self.lr_metric = LRFactory(self.flags)  # OnPlateauLRPy(self.flags)

            for i in tqdm(range(start_iter, self.flags.max_iter + 1), ncols=80):
                # training
                summary_train, _, curr_loss, curr_lr = sess.run(
                    [self.summ_train, self.train_op, self.total_loss, self.lr])
                summary_writer.add_summary(summary_train, i)

                # testing
                if i % self.flags.test_every_iter == 0:
                    # run testing average
                    avg_test_dict = self.run_k_test_iterations(sess)

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
                    summary_alw, _ = sess.run([self.summ_train, self.train_op])
                else:
                    summary, _ = sess.run([self.summ_train, self.train_op],
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
        assert (self.flags.ckpt)  # the self.flags.ckpt should be provided
        tf_saver = tf.train.Saver(max_to_keep=10)
        logdir = os.path.join(self.flags.logdir, os.path.basename(self.flags.ckpt).split(".")[0])

        # read test split filenames
        filenames = open(TEST_SPLIT, "r").readlines()
        # start
        test_metrics_dict = {key: np.zeros(value.get_shape()) for key, value in self.test_tensors_dict.items()}
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.summ2txt_line(self.flags.ckpt)
            self.summ2txt_line(DELIMITER.join(['iteration'] + [key for key in self.test_tensors_dict.keys() if key != CONF_MAT_KEY]))

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

            print('Start testing ...')
            for i in range(0, self.flags.test_iter):
                iter_test_result_dict, iter_tdc = sess.run([self.test_tensors_dict, self.test_debug_checks])
                iter_test_result_dict = self.result_callback(iter_test_result_dict)

                points, labels, probabilities = iter_tdc['/pts(xyz)'][:, 0:3], iter_tdc['/label'], iter_tdc['/probabilities']

                predictions = np.argmax(probabilities, axis=1).astype(np.int32)
                prediction_colors = np.array([to_rgb(COLOURS[int(p)]) for p in predictions])

                reports = filenames[i].strip()+": "
                for key, value in iter_test_result_dict.items():
                    test_metrics_dict[key] += value
                    if key != CONF_MAT_KEY:
                        reports += '%s: %0.4f; ' % (key, value)
                print(reports)

                current_iou = int(iter_test_result_dict['iou'] * 100)
                current_ply_o_f = os.path.join(predicted_ply_dir, "iou{}_i{}.ply".format(current_iou, i))
                save_ply(current_ply_o_f, points, prediction_colors)

                np.save(file=os.path.join(probabilities_dir, filenames[i].strip()), arr=np.array(probabilities))

                self.summ2txt([value for key, value in iter_test_result_dict.items() if key != CONF_MAT_KEY], i)

        # Average test results
        for key, value in test_metrics_dict.items():
            test_metrics_dict[key] /= self.flags.test_iter
        test_metrics_dict = self.result_callback(test_metrics_dict)

        # print the results
        print('Testing done!\n')
        reports = 'ALL: %04d; ' % self.flags.test_iter
        avg_test_sorted = []
        for key in iter_test_result_dict.keys():
            if key != CONF_MAT_KEY:
                avg_test_sorted.append(test_metrics_dict[key])
                reports += '%s: %0.4f; ' % (key, test_metrics_dict[key])
            else:
                vis_confusion_matrix(os.path.join(logdir, "confusion_matrix.png"),
                                     test_metrics_dict[key].reshape(self.num_class, self.num_class),
                                     CATEGORIES,
                                     COLOURS,
                                     "Test samples: {}".format(self.flags.test_iter))
        print(reports)
        self.summ2txt(avg_test_sorted, 'ALL')


def save_pickled_np(filename, probabilities):
    pickle.dump(probabilities, open(filename, 'wb'))


# run the experiments
if __name__ == '__main__':
    t = time.time()
    FLAGS = parse_args()
    compute_graph = ComputeGraphSeg(FLAGS)
    builder_op = build_solver_given_lr if FLAGS.SOLVER.lr_type == 'plateau' else build_solver
    solver = PartNetSolver(FLAGS, compute_graph, builder_op)
    solver.run()
    print("Minutes passed {}".format((time.time() - t)/60))
