import os


current_path = os.path.dirname(os.path.realpath(__file__))
convert_tfrecords = os.path.join(current_path, '../util/convert_tfrecords.py')


if __name__ == '__main__':
  # cmds = ['python', convert_tfrecords,
  #         '--file_dir', '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/ocnn_points',
  #         '--list_file', "/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset/test_points.txt",
  #         '--records_name', "/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset/test_points.tfrecords",
  #         '']
  # cmd = ' '.join(cmds)
  # print(cmd)
  # os.system(cmd)
  #
  # cmds = ['python', convert_tfrecords,
  #         '--file_dir', '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/ocnn_points',
  #         '--list_file', "/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset/train_points.txt",
  #         '--records_name', "/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset/train_points.tfrecords",
  #         '--shuffle true']
  # cmd = ' '.join(cmds)
  # print(cmd)
  # os.system(cmd)

    cmds = ['python', convert_tfrecords,
            '--file_dir', '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/octrees_d7',
            '--list_file', "/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset/test_octrees.txt",
            '--records_name', "/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset/test_octrees.tfrecords",
            '']
    cmd = ' '.join(cmds)
    print(cmd)
    os.system(cmd)

    cmds = ['python', convert_tfrecords,
            '--file_dir', '/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/octrees_d7',
            '--list_file', "/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset/train_octrees.txt",
            '--records_name', "/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset/train_octrees.tfrecords",
            '--shuffle true']
    cmd = ' '.join(cmds)
    print(cmd)
    os.system(cmd)
