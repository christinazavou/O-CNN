Approach 1
---

given:
/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/w_colour_norm_w_labels_sample
that contains 16 files

i run:
from_colored_annotated_data_to_default_ply(
      '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/',
      'w_colour_norm_w_labels_sample',
      sample_pts=1000
  )

where this calls:

def from_colored_annotated_data_to_default_ply(root_dir, data_dir, sample_pts=None):
  ply_dir = 'ply100000' if sample_pts is None else 'ply'+str(sample_pts)
  output_path = os.path.join(root_dir, ply_dir)
  print('Convert the raw data to ply files in {}...'.format(output_path))
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  for annotated_file in os.listdir(os.path.join(root_dir, data_dir)):
    annotation_id = annotated_file.replace('_w_label.txt', '')
    from_colored_single_data_to_default_ply(os.path.join(root_dir, data_dir, annotated_file),
                                            os.path.join(output_path, annotation_id + ".ply"),
                                            sample_pts)


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

and creates a ply1000 folder with 16 ply objects that have 1000 points and look like the ones created in partnet


then i run:

convert_points('/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/',
                 sample_pts=1000)

which runs:


def convert_points(root_dir, sample_pts=None):
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
  print("filenames ", filenames)

  list_filename = os.path.join(root_dir, 'filelist_{}.txt'.format(os.path.split(ply_dir)[1]))
  print("list_filename ", list_filename)
  with open(list_filename, 'w') as fid:
    fid.write('\n'.join(filenames))

  cmd = ' '.join([
      '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/octree/build/./ply2points',
      '--filenames', list_filename,
      '--output_path', points_dir,
      '--verbose', '0'])
  print(cmd + "\n")
  os.system(cmd)

  and creates a points1000 folder with 16 points objects that have 1000 points and should look like the ones created in partnet

