import os
import sys
from random import shuffle as _shuffle

import tensorflow as tf

sys.path.append("../..")
from libs import bounding_sphere, points2octree, transform_points, octree_batch


class TransformPoints:
    def __init__(self, distort, depth, offset=0.55, axis='xyz', scale=0.25,
                 jitter=8, drop_dim=[8, 32], angle=[20, 180, 20], dropout=[0, 0],
                 stddev=[0, 0, 0], uniform=False, interval=[1, 1, 1],
                 bounding_sphere=bounding_sphere, **kwargs):
        self.distort = distort
        self.axis = axis
        self.scale = scale
        self.jitter = jitter
        self.depth = depth
        self.offset = offset
        self.angle = angle
        self.drop_dim = drop_dim
        self.dropout = dropout
        self.stddev = stddev
        self.uniform_scale = uniform
        self.interval = interval
        self.bounding_sphere = bounding_sphere

    def __call__(self, points):
        angle, scale, jitter, ratio, dim, angle, stddev = 0.0, 1.0, 0.0, 0.0, 0, 0, 0

        if self.distort:
            angle = [0, 0, 0]
            for i in range(3):
                interval = self.interval[i] if self.interval[i] > 1 else 1
                rot_num = self.angle[i] // interval
                rnd = tf.random.uniform(shape=[], minval=-rot_num, maxval=rot_num, dtype=tf.int32)
                angle[i] = tf.cast(rnd, dtype=tf.float32) * (interval * 3.14159265 / 180.0)
            angle = tf.stack(angle)

            minval, maxval = 1 - self.scale, 1 + self.scale
            scale = tf.random.uniform(shape=[3], minval=minval, maxval=maxval, dtype=tf.float32)
            if self.uniform_scale:
                scale = tf.stack([scale[0]] * 3)

            minval, maxval = -self.jitter, self.jitter
            jitter = tf.random.uniform(shape=[3], minval=minval, maxval=maxval, dtype=tf.float32)

            minval, maxval = self.dropout[0], self.dropout[1]
            ratio = tf.random.uniform(shape=[], minval=minval, maxval=maxval, dtype=tf.float32)
            minval, maxval = self.drop_dim[0], self.drop_dim[1]
            dim = tf.random.uniform(shape=[], minval=minval, maxval=maxval, dtype=tf.int32)
            # dim = tf.cond(tf.random_uniform([], 0, 1) > 0.5, lambda: 0,
            #     lambda: tf.random.uniform(shape=[], minval=minval, maxval=maxval, dtype=tf.int32))

            stddev = [tf.random.uniform(shape=[], minval=0, maxval=s) for s in self.stddev]
            stddev = tf.stack(stddev)

        radius, center = self.bounding_sphere(points)
        points = transform_points(points, angle=angle, scale=scale, jitter=jitter,
                                  radius=radius, center=center, axis=self.axis,
                                  depth=self.depth, offset=self.offset,
                                  ratio=ratio, dim=dim, stddev=stddev)
        # The range of points is [-1, 1]
        return points  # TODO: return the transformations


class Points2Octree:
    def __init__(self, depth, full_depth=2, node_dis=False, node_feat=False,
                 split_label=False, adaptive=False, adp_depth=4, th_normal=0.1,
                 save_pts=False, **kwargs):
        self.depth = depth
        self.full_depth = full_depth
        self.node_dis = node_dis
        self.node_feat = node_feat
        self.split_label = split_label
        self.adaptive = adaptive
        self.adp_depth = adp_depth
        self.th_normal = th_normal
        self.save_pts = save_pts

    def __call__(self, points):
        octree = points2octree(points, depth=self.depth, full_depth=self.full_depth,
                               node_dis=self.node_dis, node_feature=self.node_feat,
                               split_label=self.split_label, adaptive=self.adaptive,
                               adp_depth=self.adp_depth, th_normal=self.th_normal,
                               save_pts=self.save_pts)
        return octree


class ParseExampleDebug:
    def __init__(self, x_alias='data', y_alias='label', **kwargs):
        self.x_alias = x_alias
        self.y_alias = y_alias
        self.features = {x_alias: TFRecordsUtils.from_bytes_feature(),
                         y_alias: TFRecordsUtils.from_ints_feature(),
                         'index': TFRecordsUtils.from_ints_feature(),
                         'filename': TFRecordsUtils.from_bytes_feature()}

    def __call__(self, record):
        parsed = tf.io.parse_single_example(record, self.features)
        return parsed[self.x_alias], parsed[self.y_alias], parsed['filename']


class PointCloudDataset:
    def __init__(self, parse_example):
        self.parse_example = parse_example

    def __call__(self, tf_record_filenames, batch_size, shuffle_size=1000,
                 return_iterator=False, take=-1, **kwargs):
        with tf.name_scope('points_dataset'):
            def preprocess(record):
                points, label, filename = self.parse_example(record)
                return points, label, filename

            dataset = tf.data.TFRecordDataset(tf_record_filenames) \
                .take(take) \
                .repeat()
            if shuffle_size > 1:
                dataset = dataset.shuffle(shuffle_size)
            itr = dataset \
                .map(preprocess, num_parallel_calls=8) \
                .batch(batch_size) \
                .prefetch(8) \
                .make_one_shot_iterator()
        return itr if return_iterator else itr.get_next()


class Point2OctreeDataset:
    def __init__(self, parse_example, transform_points, points2octree):
        self.parse_example = parse_example
        self.transform_points = transform_points
        self.points2octree = points2octree

    def __call__(self, tf_record_filenames, batch_size, shuffle_size=1000,
                 return_iterator=False, take=-1, return_pts=False, **kwargs):
        with tf.name_scope('points_dataset'):
            def preprocess(record):
                points, label, filename = self.parse_example(record)
                points = self.transform_points(points)
                octree = self.points2octree(points)
                outputs = (octree, label)
                if return_pts:
                    outputs += (points,)
                return outputs

            def merge_octrees(octrees, *args):
                octree = octree_batch(octrees)
                return (octree,) + args

            dataset = tf.data.TFRecordDataset(tf_record_filenames) \
                .take(take) \
                .repeat()
            if shuffle_size > 1:
                dataset = dataset.shuffle(shuffle_size)
            itr = dataset \
                .map(preprocess, num_parallel_calls=8) \
                .batch(batch_size) \
                .map(merge_octrees, num_parallel_calls=8) \
                .prefetch(8) \
                .make_one_shot_iterator()
        return itr if return_iterator else itr.get_next()


class OctreeDatasetDebug:
    def __init__(self, parse_example):
        self.parse_example = parse_example

    def __call__(self, tf_record_filenames, batch_size, shuffle_size=1000,
                 return_iterator=False, take=-1, **kwargs):
        with tf.name_scope('octree_dataset'):
            def merge_octrees(octrees, labels, filenames):
                return octree_batch(octrees), labels, filenames

            dataset = tf.data.TFRecordDataset(tf_record_filenames) \
                .take(take) \
                .repeat()
            if shuffle_size > 1:
                dataset = dataset.shuffle(shuffle_size)
            itr = dataset \
                .map(self.parse_example, num_parallel_calls=8) \
                .batch(batch_size) \
                .map(merge_octrees, num_parallel_calls=8) \
                .prefetch(8) \
                .make_one_shot_iterator()
        return itr if return_iterator else itr.get_next()


class DatasetFactoryDebug:
    def __init__(self, flags):
        self.flags = flags
        if flags.dtype == 'octree':
            self.dataset = OctreeDatasetDebug(ParseExampleDebug(**flags))
        else:
            raise Exception('Error: unsupported datatype ' + flags.dtype)

    def __call__(self):
        return self.dataset(tf_record_filenames=self.flags.location,
                            batch_size=self.flags.batch_size,
                            shuffle_size=self.flags.shuffle,
                            return_iterator=self.flags.return_iterator,
                            take=self.flags.n_samples,
                            return_pts=self.flags.return_pts)


class TFRecordsUtils:
    @staticmethod
    def to_int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def to_bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def from_bytes_feature():
        return tf.FixedLenFeature([], tf.string)

    @staticmethod
    def from_ints_feature():
        return tf.FixedLenFeature([], tf.int64)

    @staticmethod
    def byte_from_example(example, feature_name):
        return example.features.feature[feature_name].bytes_list.value[0]

    @staticmethod
    def int_from_example(example, feature_name):
        return int(example.features.feature[feature_name].int64_list.value[0])


class TFRWriter:
    def __init__(self, records_name):
        self.writer = tf.python_io.TFRecordWriter(records_name)

    def __call__(self, file_type, octrees_dir, label, index, in_filename, save_filename):
        octree_file = TFRecordsConverter.load_octree(os.path.join(octrees_dir, in_filename))
        feature = {file_type: TFRecordsUtils.to_bytes_feature(octree_file),
                   'label': TFRecordsUtils.to_int64_feature(label),
                   'index': TFRecordsUtils.to_int64_feature(index),
                   'filename': TFRecordsUtils.to_bytes_feature(save_filename)}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(example.SerializeToString())

    def close(self):
        self.writer.close()


class TFRReader:

    def __init__(self, records_name):
        self.records_iterator = tf.python_io.tf_record_iterator(records_name)

    def __call__(self, take_max, file_type='data'):
        num = 0
        for string_record in self.records_iterator:
            if num >= take_max:
                break

            example = tf.train.Example()
            example.ParseFromString(string_record)
            label = TFRecordsUtils.int_from_example(example, 'label')
            index = TFRecordsUtils.int_from_example(example, 'index')
            octree = TFRecordsUtils.byte_from_example(example, file_type)
            if 'filename' in example.features.feature:
                filename = TFRecordsUtils.byte_from_example(example, 'filename') \
                    .decode('utf8').replace('/', '_').replace('\\', '_')
            else:
                filename = '%06d.%s' % (num, file_type)
            num += 1
            yield octree, label, index, filename


class TFRecordsConverter:

    @staticmethod
    def load_octree(file):
        with open(file, 'rb') as f:
            octree_bytes = f.read()
        return octree_bytes

    @staticmethod
    def write_records(octrees_dir, list_file, records_name, file_type='data', shuffle=False):
        [data, label, index] = TFRecordsConverter.get_data_label_pair(list_file, shuffle)

        if not os.path.exists(os.path.dirname(records_name)):
            os.makedirs(os.path.dirname(records_name))

        writer = TFRWriter(records_name)
        for i in range(len(data)):
            if not i % 1000:
                print('data loaded: {}/{}'.format(i, len(data)))
            writer(file_type,
                   octrees_dir,
                   label[i],
                   index[i],
                   data[i],
                   ('%06d_%s' % (i, data[i])).encode('utf8'))
        writer.close()

    @staticmethod
    def get_data_label_pair(list_file, shuffle_data=False):
        filepath_list = []
        label_list = []
        with open(list_file) as f:
            for line in f:
                filepath, label = line.split()
                filepath_list.append(filepath)
                label_list.append(int(label))
        index_list = list(range(len(label_list)))

        if shuffle_data:
            c = list(zip(filepath_list, label_list, index_list))
            _shuffle(c)
            filepath_list, label_list, index_list = zip(*c)
            with open(list_file + '.shuffle.txt', 'w') as f:
                for item in c:
                    f.write('{} {}\n'.format(item[0], item[1]))
        return filepath_list, label_list, index_list

    @staticmethod
    def read_records(records_name, output_path, list_file, file_type='data', count=0):
        reader = TFRReader(records_name)
        count = count if count != 0 else float('Inf')

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, list_file), "w") as f:
            for octree, label, index, filename in reader(count, file_type):
                with open(os.path.join(output_path, filename), 'wb') as fo:
                    fo.write(octree)
                f.write("{} {}\n".format(filename, label))


class ClassificationDatasetStatistics:

    def __init__(self, points_folder):
        self.folder = points_folder
        self.categories = os.listdir(points_folder)
        self.test_samples, self.train_samples = self.points_per_class()
        self.train_n = sum(self.train_samples)
        self.test_n = sum(self.test_samples)

    def points_per_class(self):
        test_samples = []
        train_samples = []
        for category in self.categories:
            directory = os.path.join(self.folder, category, "test")
            points_filenames = [f for f in os.listdir(directory) if '.points' in f]
            test_samples.append(len(points_filenames))

            directory = os.path.join(self.folder, category, "train")
            points_filenames = [f for f in os.listdir(directory) if '.points' in f]
            train_samples.append(len(points_filenames))
        return test_samples, train_samples


class FileManipulator:

    @staticmethod
    def generate_list_text_files(points_folder):
        for point_folder in os.listdir(points_folder):
            point_folder = os.path.join(points_folder, point_folder)
            print("processing point folder: ", point_folder)
            with open(os.path.join(point_folder, "list.txt"), "w") as f:
                for point_name in os.listdir(point_folder):
                    if ".points" in point_name:
                        point_file = os.path.join(point_folder, point_name)
                        f.write(point_file + "\n")

    @staticmethod
    def generate_octrees_for_each_folder(points_folder, out_dir, args):
        for point_folder in os.listdir(points_folder):
            filenames = os.path.join(points_folder, point_folder, "list.txt")
            output_path = os.path.join(out_dir, point_folder)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            cmd = "cd /home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/octree/build && ./octree --filenames {} --output_path {} {}".format(
                filenames, output_path, args)
            print(cmd)
            os.system(cmd)

    @staticmethod
    def point_list_to_octree_list(point_file, octree_file, depth, rot_num, full_depth=2):
        with open(point_file, "r") as fin, open(octree_file, "w") as fout:
            lines = fin.readlines()
            for line in lines:
                for i in range(rot_num):
                    new_prefix = "_{}_{}_".format(depth, full_depth) + "{0:03}.octree".format(i)
                    fout.write(line.replace(".points", new_prefix))


if __name__ == '__main__':
    try:
        eval(sys.argv[1])
    except Exception as e:
        print("Couldn't evaluate and run the given command: " + sys.argv[1])
        raise Exception(e)
