import sys
import os
import numpy as np
import json
import psutil
from tqdm import tqdm


def normalise_colour(pts_colour):
    pts_colour = np.true_divide(pts_colour, 255.0)
    return pts_colour


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

        def read_files(flags, nout):
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
                        if pts.shape[-1] > 6:
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

        read_files(flags, nout)

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
            self.features=np.zeros((len(self.filenames),1))

        self.points = np.asarray(self.points).astype(dtype=np.float32)
        self.normals = np.asarray(self.normals).astype(dtype=np.float32)
        self.point_labels = np.asarray(self.point_labels)
        self.filenames = np.asarray(self.filenames).astype(dtype="str")

        if mem_check:
            # check memory consumption
            py = psutil.Process(os.getpid())
            memory_usage = py.memory_info()[0] / 1024 ** 3
            print("memory: ", memory_usage)

        return self
