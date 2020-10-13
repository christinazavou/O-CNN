import numpy as np
import math
import argparse
import os
import multiprocessing as mp


# code from: https://github.com/charlesq34/pointnet2/blob/74bb67f3702e8aec55a7b8765dd728b18456030c/utils/provider.py
def rotate_point_cloud_by_angle(points, angle=0.0):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array, original point clouds
          scalar, angle of rotation in radians

        Return:
          Nx3 array, rotated point clouds
    """
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    points[0:3] = np.dot(points[0:3].reshape((-1, 3)), rotation_matrix)
    return points


def rotate_point_cloud_by_angle_with_normal(points, angle=0.0):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx6 array, original point clouds with normal
          scalar, angle of rotation in radians
        Return:
          Nx6 array, rotated point clouds with normal
    """
    cosval = np.cos(angle)
    sinval = np.sin(angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])

    points[..., 0:3] = np.dot(points[..., 0:3].reshape((-1, 3)), rotation_matrix)
    points[..., 3:6] = np.dot(points[..., 3:6].reshape((-1, 3)), rotation_matrix)
    return points


def save_rot(func, data, id):
    startidx = 0
    if os.path.exists(os.path.join(LOG_DIR, "cur_model_{:03d}.txt".format(id))):
        startidx = int(open(os.path.join(LOG_DIR, "cur_model_{:03d}.txt".format(id)), "r").read())

    d = (2 * math.pi) / float(ARGS.rot_num)

    print("Starting point cloud rotation process {procID:d}...".format(procID=id))
    for i in range(startidx, len(data)):
        filename = data[i].strip()
        if ".ply" in filename:
            lines = open(filename, "r").readlines()
            while not "end_header" in lines[0]:
                lines.pop(0)
            lines.pop(0)
            open("./tmp_{:03d}.txt".format(id), "w").writelines(lines)
            pc = np.loadtxt("./tmp_{:03d}.txt".format(id))
        else:
            pc = np.loadtxt(filename)
        fname = os.path.basename(filename).split(".")[0]
        np.savetxt(os.path.join(ARGS.output, fname + "_000.txt"), pc)

        for j in range(1, ARGS.rot_num):
            rot_pc = func(pc, j * d)
            np.savetxt(os.path.join(ARGS.output, fname + "_{:03d}.txt".format(j)), rot_pc)
        open(os.path.join(LOG_DIR, "cur_model_{:03d}.txt".format(id)), "w").write(str(i+1))
        print("Process {procID:d}: Processed files ({nProc:d}/{nModels:d})".format(procID=id, nProc=i + 1,
                                                                                   nModels=len(data)))
    print("Terminating process {procID:d}".format(procID=id))


def runParallel(func, data):
    # Number of files to be processed by each process
    step = int(round(len(data) / float(ARGS.n_proc)))
    proc = []
    for i in range(ARGS.n_proc):
        startIdx = i * step
        if (ARGS.n_proc - 1) > i:
            endIdx = (i + 1) * step
        else:
            endIdx = len(data)
        p = mp.Process(target=save_rot, args=(func, data[startIdx:endIdx], i))
        proc.append(p)
        p.start()

    # Wait for processes to end
    for oneProc in proc:
        oneProc.join()
    for i in range(ARGS.rot_num):
        if os.path.exists("./tmp_{:03d}.txt".format(i)):
            os.remove("./tmp_{:03d}.txt".format(i))


parser = argparse.ArgumentParser(description="Rotate point cloud a given number of times.")
parser.add_argument("--rot_num", default=12, type=int, help="Number of times to rotate points. Rotation "
                                                            "angle=360/rot_num.")
parser.add_argument("--n_proc", default=12, type=int, help="Number of parallel processes to run. Default: 12.")
parser.add_argument("-wn", "--with_normals", type=bool, default=True, help="Whether input has normal info.")
parser.add_argument("-pc", "--point_cloud", required=True, help="File/Directory with point clouds to rotate.")
parser.add_argument('-o', "--output", default="./",
                    help="Directory to save rotated points. Default: same as point clouds.")
ARGS = parser.parse_args()

LOG_DIR = "log"
if __name__ == "__main__":
    if ARGS.rot_num <= 0:
        print("{} is an invalid number of rotations. Exiting...".format(ARGS.rot_num))
        exit()
    if not os.path.exists(ARGS.point_cloud):
        print("Point cloud path doesn't exist. Exiting...")
        exit()
    if ARGS.output == "./":
        ARGS.output = os.path.join(ARGS.point_cloud, "Rotated_points")
        if not os.path.exists(ARGS.output):
            os.mkdir(ARGS.output)
    elif not os.path.exists(ARGS.output) or not os.path.isdir(ARGS.output):
        print("Output path doesn't exist or it is not a directory. Exiting...")
        exit()
    pc = []
    if os.path.isdir(ARGS.point_cloud):
        for root, _, files in os.walk(ARGS.point_cloud):
            for file in files:
                pc.append(os.path.join(root, file))
    else:
        pc=open(ARGS.point_cloud,"r").readlines()

    LOG_DIR = os.path.join(ARGS.output, LOG_DIR)
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)
    rot_op = rotate_point_cloud_by_angle_with_normal if ARGS.with_normals else rotate_point_cloud_by_angle
    runParallel(rot_op, pc)
    print("Done.")
