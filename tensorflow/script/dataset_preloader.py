import sys
import os
import numpy as np
import json
import psutil
from tqdm import tqdm
from math import ceil
from colorsys import rgb_to_hsv
from warnings import warn


def normalise_colour(pts_colour):
    pts_colour = np.true_divide(pts_colour, 255.0)
    return pts_colour


def rgb2hsv(rgb):
    return rgb_to_hsv(rgb[0], rgb[1], rgb[2])


def colour_convertor(fts):
    fts = np.delete(fts, 3, 2)  # drop alpha channel since it is not used in HSV
    if np.any(fts > 1):  # rgb_to_hsv expects colour values from 0 to 1
        fts = normalise_colour(fts)
    fts = np.apply_along_axis(rgb2hsv, 2, fts)
    return fts


class DataLoader:
    def __init__(self):
        self.points = []
        self.normals = []
        self.features = []
        self.filenames = []
        self.point_labels = []
        self.tfrecord_num = 0
        self.CHUNK_SIZE = 100

    def __call__(self, flags, nout, channels, mem_check=False, test_phase=False):

        def load_points_file(filename):
            if ".ply" in filename:
                try:
                    p = open(filename, "r").readlines()
                except OSError:
                    print("Could not open file: ", filename)
                    sys.exit()

                while not "end_header" in p[0]:
                    p.pop(0)
                p.pop(0)
                point_clouds = np.array([[float(i) for i in j.strip().split()] for j in p])
            else:  # txt
                try:
                    point_clouds = np.loadtxt(filename).astype(np.float32)
                except ValueError:
                    print("Could not load file: ", filename)
                    sys.exit()

            return point_clouds

        def read_files(filenames, nout, channels):
            points = []
            normals = []
            features = []
            point_labels = []

            try:
                for fname in tqdm(filenames,leave=False,file=sys.stdout):
                    pts = load_points_file(os.path.join(self.flags.location, fname))
                    points.append(pts[..., 0:3])
                    normals.append(pts[..., 3:6])

                    # check for available point features
                    if channels > 4:
                        if pts.shape[-1] <= 6:  # x,y,z,nx,ny,nz,fts
                            print("Point features are not available. Exiting...")
                            sys.exit()
                        else:
                            features.append(pts[..., 6:])

                    labels = np.array(list(json.load(open(
                        os.path.join(self.flags.label_location, fname.split(".")[0] + "_label.json"))).values()),
                                      dtype=np.float32)
                    # map labels so that learnable ones start from 0 and undetermined becomes num_classes+1 (to
                    # be ignored in loss)
                    labels = np.where(labels == 0, nout, labels - 1)
                    point_labels.append(labels)

            except OSError:
                print("Could not open data file list. Exiting...")
                sys.exit()

            return np.asarray(points), np.asarray(normals), np.asarray(features), np.asarray(point_labels)

        self.flags = flags
        self.filenames = open(self.flags.file_list, "r").readlines()
        self.tfrecord_num = len(self.filenames)

        if self.flags.take != -1:
            if self.flags.take >= self.tfrecord_num:
                warn("Flag take is larger or equal to the number of records ({} vs {}). Ignoring flag take".format(
                    self.flags.take, self.tfrecord_num))
            else:
                warn(
                    "Flag take is not -1. Ignoring {} last records from file".format(
                        self.flags.take))
                del self.filenames[(self.tfrecord_num - self.flags.take):]
                self.tfrecord_num -= self.flags.take

        self.filenames = [line.strip().split()[0] for line in self.filenames]

        # split file reads to chunks for less memory usage
        reps = ceil(self.tfrecord_num / self.CHUNK_SIZE)
        print("Spliting data into {} chunks.".format(reps))
        # read first chunk
        print("Reading chunk: 1/{}".format(reps))
        self.points, self.normals, self.features, self.point_labels = read_files(
            self.filenames[:min(self.tfrecord_num, self.CHUNK_SIZE)], nout, channels)

        if channels > 4:  # extra features besides normals
            if self.features.shape[-1] != channels - 3 - (1 if self.flags.node_dis else 0):
                raise ValueError(
                    "Number of features in input files and MODEL.channel parameter don't agree ({} vs {})".format(
                        self.features.shape[-1] + 3 + (1 if self.flags.node_dis else 0), channels))
        else:
            self.features = np.zeros((len(self.filenames), 1))

        for c in range(1, reps):
            print("Reading chunk: {}/{}".format(c + 1, reps))
            p, n, f, l = read_files(
                self.filenames[c * self.CHUNK_SIZE:min((c + 1) * self.CHUNK_SIZE, self.tfrecord_num)], nout, channels)
            self.points = np.append(self.points, p, axis=0)
            self.normals = np.append(self.normals, n, axis=0)
            self.point_labels = np.append(self.point_labels, l, axis=0)
            if f.size:
                self.features = np.append(self.features, f, axis=0)
        if test_phase:
            self.filenames = [line.split(".")[0] for line in self.filenames]
            self.filenames = np.asarray(self.filenames).astype(dtype="str")
        else:
            del self.filenames

        if channels > 4 and self.flags.hsv:
            self.features = colour_convertor(self.features)

        if mem_check:
            # check memory consumption
            py = psutil.Process(os.getpid())
            memory_usage = py.memory_info()[0] / 1024 ** 3
            print("memory: ", memory_usage)

        return self
