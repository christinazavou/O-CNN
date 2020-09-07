import os
import sys

import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
convert_tfrecords = os.path.join(current_path, '../util/convert_tfrecords.py')


def from_colored_annotated_data_to_default_ply(root_dir, data_dir, sample_pts=None, parallel=False):
    ply_dir = 'ply100000' if sample_pts is None else 'ply' + str(sample_pts)
    output_path = os.path.join(root_dir, ply_dir)
    print('Convert the raw data to ply files in {}...'.format(output_path))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not parallel:
        not_parallel(root_dir, data_dir, output_path, sample_pts)
    else:
        run_parallel(root_dir, data_dir, output_path, sample_pts)


def not_parallel(root_dir, data_dir, output_path, sample_pts):
    for annotated_file in os.listdir(os.path.join(root_dir, data_dir)):
        annotation_id = annotated_file.replace('_w_label.txt', '')
        from_colored_single_data_to_default_ply(os.path.join(root_dir, data_dir, annotated_file),
                                                os.path.join(output_path, annotation_id + ".ply"),
                                                sample_pts)


def run_parallel(root_dir, data_dir, output_path, sample_pts):
    import multiprocessing as mp
    pool = mp.Pool(6)
    results = [pool.apply(from_colored_single_data_to_default_ply,
                          args=(os.path.join(root_dir, data_dir, annotated_file),
                                os.path.join(output_path, annotated_file.replace('_w_label.txt', '') + ".ply"),
                                sample_pts))
               for annotated_file in os.listdir(os.path.join(root_dir, data_dir))]
    pool.close()
    print("results", results)


def from_colored_single_data_to_default_ply(input_file, output_file, sample_pts=None):
    a = np.loadtxt(input_file).astype(np.float32)
    if sample_pts:
        sample_indices = np.random.randint(0, a.shape[0], sample_pts)
        points = a[sample_indices, 0:3]
        normals = a[sample_indices, 3:6]
        features = a[sample_indices, 6:10]
        labels = a[sample_indices, 10].astype(np.int)
    else:
        points = a[:, 0:3]
        normals = a[:, 3:6]
        features = a[:, 6:10]
        labels = a[:, 10].astype(np.int)
    save_ply(output_file, points, normals, labels)


def save_ply(filename, points, normals, labels):
    assert points.shape[0] == normals.shape[0] == labels.shape[0]
    assert points.shape[1] == normals.shape[1] == 3
    labels = labels.reshape((points.shape[0], 1))
    data = np.concatenate([points, normals, labels], axis=1)

    header = "ply\nformat ascii 1.0\nelement vertex %d\n" \
             "property float x\nproperty float y\nproperty float z\n" \
             "property float nx\nproperty float ny\nproperty float nz\n" \
             "property float label\nelement face 0\n" \
             "property list uchar int vertex_indices\nend_header\n"
    with open(filename, 'w') as fid:
        fid.write(header % points.shape[0])
        np.savetxt(fid, data, fmt='%.6f')


def chunks(l, n):
    return (l[x: x + n] for x in range(0, len(l), n))


def convert_points(root_dir, sample_pts=None, parallel=False):
    ply_dir = os.path.join(root_dir, 'ply100000' if sample_pts is None else 'ply' + str(sample_pts))
    suffix = 'points100000' if sample_pts is None else 'points' + str(sample_pts)
    points_dir = os.path.join(root_dir, suffix)
    print('Convert ply files to points files in {}...'.format(points_dir))
    if not os.path.exists(points_dir):
        os.makedirs(points_dir)

    filenames = []
    for filename in os.listdir(ply_dir):
        if filename.endswith('.ply'):
            filenames.append(os.path.join(ply_dir, filename))

    if not parallel:
        not_parallel_points(root_dir, filenames, ply_dir, points_dir)
    else:
        parallel_points__(root_dir, filenames, ply_dir, points_dir)


def parallel_points__(root_dir, filenames, ply_dir, points_dir):
    list_filenames = []

    for chunk, files in enumerate(chunks(filenames, int(len(filenames) / 6))):
        list_filename = os.path.join(root_dir, 'filelist_{}_chunk{}.txt'.format(os.path.split(ply_dir)[1], chunk))
        with open(list_filename, 'w') as fid:
            fid.write('\n'.join(files))
        list_filenames.append(list_filename)

    for list_f in list_filenames:
        command = ' '.join(['/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/octree/build/./ply2points',
                            '--filenames', list_f,
                            '--output_path', points_dir,
                            '--verbose', '0'])
        print(command)
    print("YOU SHOULD RUN THEM MANUALLY")


def one_command(list_filename, points_dir):
    cmd = ' '.join([
        '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/octree/build/./ply2points',
        '--filenames', list_filename,
        '--output_path', points_dir,
        '--verbose', '0'])
    print(cmd + "\n")
    os.system(cmd)


def not_parallel_points(root_dir, filenames, ply_dir, points_dir):
    list_filename = os.path.join(root_dir, 'filelist_{}.txt'.format(os.path.split(ply_dir)[1]))
    with open(list_filename, 'w') as fid:
        fid.write('\n'.join(filenames))

    one_command(list_filename, points_dir)


def convert_points_to_tfrecords(root_dir, records_dir, sample_pts=None):
    points_dir = 'points100000' if sample_pts is None else 'points' + str(sample_pts)

    for phase in ["val", "test", "train"]:
        output_record = "{}_{}.tfrecords".format(phase, points_dir)
        _points_dir = os.path.join(root_dir, points_dir)
        output_record = os.path.join(root_dir, records_dir, output_record)
        filelist_in = os.path.join(root_dir, records_dir, "{}_points.txt".format(phase))

        shuffle = '--shuffle true' if phase == 'train' else ''
        cmds = ['python', convert_tfrecords,
                '--file_dir', _points_dir,
                '--list_file', filelist_in,
                '--records_name', output_record,
                shuffle]

        cmd = ' '.join(cmds)
        print(cmd)
        os.system(cmd)


if __name__ == '__main__':
    eval(sys.argv[1])
