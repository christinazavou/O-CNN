import os
import json
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
convert_tfrecords = os.path.join(current_path, '../util/convert_tfrecords.py')

all_categoty = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher',
                'Display', 'Door', 'Earphone', 'Faucet', 'Hat', 'Keyboard',
                'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator',
                'Scissors', 'StorageFurniture', 'Table', 'TrashCan', 'Vase']
finegrained_category = ['Bed', 'Bottle', 'Chair', 'Clock', 'Dishwasher', 'Display',
                        'Door', 'Earphone', 'Faucet', 'Knife', 'Lamp', 'Microwave',
                        'Refrigerator', 'StorageFurniture', 'Table', 'TrashCan', 'Vase']
level_list_list = [[1], [1, 2, 3], [1, 3], [1], [1, 2, 3], [1, 3], [1, 2, 3],
                   [1, 3], [1, 2, 3], [1, 3], [1, 3], [1], [1], [1, 3],
                   [1, 2, 3], [1], [1, 2, 3], [1], [1, 2, 3], [1], [1, 2, 3],
                   [1, 2, 3], [1, 3], [1, 3]]
finest_level = [1, 3, 3, 1, 3, 3, 3, 3, 3, 3,
                3, 1, 1, 3, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3]
finest_level_dict = dict(zip(all_categoty, finest_level))
level_list_dict = dict(zip(all_categoty, level_list_list))


def from_colored_annotated_data_to_default_ply(sample_pts=None):
  data_path = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/w_colour_norm_w_labels'
  suffix = 'ply100000' if sample_pts is None else 'ply'+str(sample_pts)
  output_path = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/'+suffix
  print('Convert the raw data to ply files ...')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  for annotated_file in os.listdir(data_path):
    annotation_id = annotated_file.replace('_w_label.txt', '')
    from_colored_single_data_to_default_ply(os.path.join(data_path, annotated_file),
                                            os.path.join(output_path, annotation_id + ".ply"),
                                            sample_pts)


def from_colored_single_data_to_default_ply(input_file, output_file, sample_pts=None):
  a = np.loadtxt(input_file)
  if sample_pts:
    sample_indices = np.random.randint(0, a.shape[0], sample_pts)
    points = a[sample_indices, 0:3]
    normals = a[sample_indices, 3:6]
    labels = a[sample_indices, 10].astype(np.int)
  else:
    points = a[:, 0:3]
    normals = a[:, 3:6]
    labels = a[:, 10].astype(np.int)
  save_ply(output_file, points, normals, labels)


def convert_points(sample_pts=None, max_files=-1):
  root_folder = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour'
  input_folder = os.path.join(root_folder, 'ply100000' if sample_pts is None else 'ply' + str(sample_pts))
  suffix = 'points100000' if sample_pts is None else 'points' + str(sample_pts)
  suffix = suffix if max_files == -1 else str(max_files) + suffix
  output_folder = os.path.join(root_folder, suffix)
  print('Convert ply files to points files ...')
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  filenames = []
  for filename in os.listdir(input_folder):
    if filename.endswith('.ply'):
      filenames.append(os.path.join(input_folder, filename))
    if max_files != -1 and len(filenames) >= max_files:
        break

  chunk = 0
  list_filenames = []
  cmds = []
  for filenames_chunk in chunks(filenames, int(len(filenames)/4)):
    print(chunk)
    chunk += 1
    list_filename = os.path.join(output_folder, 'filelist_ply_chunk{}.txt'.format(chunk) )
    list_filenames.append(list_filename)
    with open(list_filename, 'w') as fid:
      fid.write('\n'.join(filenames_chunk))

    cmds.append(' '.join(['/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/octree/build/./ply2points',
          '--filenames', list_filename,
          '--output_path', output_folder,
          '--verbose', '0']))
  for cmd in cmds:
    print(cmd + "\n")


def convert_points_to_tfrecords(sample_pts=None, max_files=-1):
  root_folder = '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour'
  suffix = 'points100000' if sample_pts is None else 'points' + str(sample_pts)
  suffix = suffix if max_files == -1 else str(max_files) + suffix
  data_path = os.path.join(root_folder, suffix)

  for phase in ["train", "test"]:
    output_record = "{}_{}.tfrecords".format(phase, suffix)
    output_record = os.path.join(root_folder, "dataset_points", output_record)
    filelist_in = os.path.join(root_folder, "dataset_points", "_{}_points.txt".format(phase))
    filelist_out = os.path.join(root_folder, "dataset_points", "_{}_{}.txt".format(phase, suffix))

    with open(filelist_in, "r") as fin, open(filelist_out, "w") as fout:
      for line in fin.readlines():
        filename, label = line.split(" ")
        filename = os.path.split(filename)[1]
        if os.path.exists(os.path.join(data_path, filename)):
          fout.write(filename + " " + label)

    shuffle = '--shuffle true' if phase == 'train' else ''
    cmds = ['python', convert_tfrecords,
              '--file_dir', data_path,
              '--list_file', filelist_out,
              '--records_name', output_record,
              shuffle]

    cmd = ' '.join(cmds)
    print(cmd)
    os.system(cmd)


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
  n = max(1, n)
  return (l[i:i+n] for i in range(0, len(l), n))


if __name__ == '__main__':
  # from_colored_annotated_data_to_default_ply()  #todo: run in parallel..
  # convert_points()
  convert_points_to_tfrecords(max_files=200)
