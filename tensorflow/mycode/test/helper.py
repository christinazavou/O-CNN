import numpy as np
from yacs.config import CfgNode as CN

from src.tf_layer_utils import *


def mock_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # image_index = 7777
    # plt.imshow(x_train[image_index], cmap='Greys')
    # plt.title("label {}".format(y_train[image_index]))
    # plt.show()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train = np.divide(x_train, 255)
    x_test = np.divide(x_test, 255)

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    print("x_train ", x_train.shape, " x_test ", x_test.shape)
    print("y_train ", y_train.shape, " y_test ", y_test.shape)
    return x_train, y_train, x_test, y_test


def mock_graph(image, num_classes):
    with tf.variable_scope('Layer1'):
        l1 = convolution_layer(image, 3, 1, 28, 1)
        l1 = pool_layer(l1, 2, 2)
        l1 = activation_layer(l1, 'relu')
        print("layer 1:", l1.get_shape())

    with tf.variable_scope('EmbeddingLayer'):
        embeddings, embeddings_length = flat_layer(l1)
        print("embeddings: ", embeddings.get_shape())

        h1 = fc_layer(embeddings, embeddings_length, 128)
        h1 = activation_layer(h1, 'relu')
        h1 = dropout_layer(h1, 0.2)
        print("h1: ", h1.get_shape())

    with tf.variable_scope('OutLayer'):
        o = fc_layer(h1, 128, num_classes)
        return o


def _mock_config():
    _C = CN()

    # DATA related parameters
    _C.DATA = CN()
    _C.DATA.train = CN()
    _C.DATA.train.dtype = 'octree'  # The data type: points or octree
    _C.DATA.train.x_alias = 'data'  # The alias of the data
    _C.DATA.train.y_alias = 'label'  # The alias of the target

    _C.DATA.train.depth = 5  # The octree depth
    _C.DATA.train.full_depth = 2  # The full depth
    _C.DATA.train.node_dis = False  # Save the node displacement
    _C.DATA.train.split_label = False  # Save the split label
    _C.DATA.train.adaptive = False  # Build the adaptive octree
    _C.DATA.train.node_feat = False  # Calculate the node feature

    _C.DATA.train.distort = False  # Whether to apply data augmentation
    _C.DATA.train.offset = 0.55  # Offset used to displace the points
    _C.DATA.train.axis = 'y'  # Rotation axis for data augmentation
    _C.DATA.train.scale = 0.0  # Scale the points
    _C.DATA.train.uniform = False  # Generate uniform scales
    _C.DATA.train.jitter = 0.0  # Jitter the points
    _C.DATA.train.drop_dim = (8, 32)  # The value used to dropout points
    _C.DATA.train.dropout = (0, 0)  # The dropout ratio
    _C.DATA.train.stddev = (0, 0, 0)  # The standard deviation of the random noise
    _C.DATA.train.interval = (1, 1, 1)  # Use interval&angle to generate random angle
    _C.DATA.train.angle = (180, 180, 180)

    _C.DATA.train.location = '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/mycode/test/resources/ModelNetOnly4Samples3/m40_5_2_12_train_octree.tfrecords'
    _C.DATA.train.shuffle = 1000  # The shuffle size
    _C.DATA.train.n_samples = 1000  # Use at most `n_samples` elements from this dataset
    _C.DATA.train.batch_size = 32  # Training data batch size
    _C.DATA.train.mask_ratio = 0.0  # Mask out some point features

    _C.DATA.train.return_iterator = False  # Return the data iterator
    _C.DATA.train.return_pts = False  # Also return points

    _C.DATA.test = _C.DATA.train.clone()
    _C.DATA.test.location = '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/mycode/test/resources/ModelNetOnly4Samples3/m40_5_2_12_test_octree.tfrecords'

    # MODEL related parameters
    _C.MODEL = CN()
    _C.MODEL.gpu = (0,)  # The gpu ids
    _C.MODEL.test_every_iter = 500  # Test model every n training steps
    _C.MODEL.learning_rate = 0.1  # Initial learning rate
    _C.MODEL.gamma = 0.1  # Learning rate step-wise decay
    _C.MODEL.lr_type = 'step'
    _C.MODEL.step_size = (40000,)  # Learning rate step size.
    _C.MODEL.ckpt_num = 100  # The number of checkpoint kept
    _C.MODEL.depth = 5  # The input octree depth
    _C.MODEL.depth_out = 5  # The output feature depth
    _C.MODEL.channel = 3  # The input feature channel
    _C.MODEL.nout = 40  # The output feature channel
    _C.MODEL.nouts = 40,  # The output feature channels
    _C.MODEL.dropout = (0.0,)  # The dropout ratio
    _C.MODEL.signal_abs = False  # Use the absolute value of signal
    _C.MODEL.upsample = 'nearest'  # The method used for upsampling
    _C.MODEL.num_class = 40  # The class number for the cross-entropy loss
    _C.MODEL.weight_decay = 0.0005  # The weight decay on model weights
    _C.MODEL.weights = (1.0, 1.0)  # The weight factors for different losses
    _C.MODEL.label_smoothing = 0.0  # The factor of label smoothing

    return _C


def mock_train_config():
    _C = _mock_config()
    _C.MODEL.logdir = '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/mycode/test/output/ModelNetOnly4Samples3/logs'  # Directory where to write event logs
    _C.MODEL.ckpt = ''  # Restore weights from checkpoint file
    _C.MODEL.run = 'train'  # Choose from train or test
    _C.MODEL.max_iter = 5000  # Maximum training iterations
    _C.MODEL.test_iter = 30  # Test steps in testing phase
    return _C


def mock_test_config():
    _C = _mock_config()
    _C.MODEL.logdir = '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/mycode/test/output/ModelNetOnly4Samples3/logs'  # Directory where to write event logs
    _C.MODEL.ckpt = 1500  # Restore weights from checkpoint file
    _C.MODEL.run = 'test'  # Choose from train or test
    _C.MODEL.max_iter = 5000  # Maximum training iterations
    _C.MODEL.test_iter = 30  # Test steps in testing phase

    return _C
