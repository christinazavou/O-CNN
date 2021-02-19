import sys
import os
import numpy as np
import json
import psutil
from tqdm import tqdm
from math import ceil
from colorsys import rgb_to_hsv


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

    def __call__(self, flags, nout, channels, mem_check=False):

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

        def read_files(flags, nout, channels):
            print("\nReading files. This may take a while...\n")
            try:
                with open(flags.file_list, "r") as f:
                    for line in tqdm(f):
                        line = line.strip().split()
                        self.filenames.append(line[0].split(".")[0])
                        pts = load_points_file(os.path.join(flags.location, line[0]))
                        self.points.append(pts[..., 0:3])
                        self.normals.append(pts[..., 3:6])

                        # check for available point features
                        if channels > 3:
                            if pts.shape[-1] <= 6:  # x,y,z,nx,ny,nz,fts
                                print("Point features are not available. Exiting...")
                                sys.exit()
                            else:
                                self.features.append(pts[..., 6:])

                        labels = np.array(list(json.load(open(
                            os.path.join(flags.label_location, line[0].split(".")[0] + "_label.json"))).values()),
                                          dtype=np.float32)
                        # map labels so that learnable ones start from 0 and undetermined becomes num_classes+1 (to
                        # be ignored in loss)
                        labels = np.where(labels == 0, nout, labels - 1)
                        self.point_labels.append(labels)
                    self.tfrecord_num = len(self.points)
            except OSError:
                print("Could not open data file list. Exiting...")
                sys.exit()

        read_files(flags, nout, channels)

        self.flags = flags
        if channels > 3:  # extra features besides normals
            self.features = np.asarray(self.features).astype(dtype=np.float32)
            if self.features.shape[-1] != channels - 3:
                raise ValueError(
                    "Number of features in input files and MODEL.channel parameter don't agree ({} vs {})".format(
                        self.features.shape[-1] + 3, channels))
            if np.any(self.features > 1):
                print("Normalising point colours...")
                self.features = normalise_colour(self.features)
        else:
            self.features = np.zeros((len(self.filenames), 1))

        self.points = np.asarray(self.points).astype(dtype=np.float32)
        self.normals = np.asarray(self.normals).astype(dtype=np.float32)
        self.point_labels = np.asarray(self.point_labels)
        self.filenames = np.asarray(self.filenames).astype(dtype="str")

        if channels > 3 and self.flags.hsv:
            self.features = colour_convertor(self.features)

        if mem_check:
            # check memory consumption
            py = psutil.Process(os.getpid())
            memory_usage = py.memory_info()[0] / 1024 ** 3
            print("memory: ", memory_usage)

        return self
