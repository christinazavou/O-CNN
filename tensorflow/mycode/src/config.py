from yacs.config import CfgNode as CN

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

_C.DATA.train.location = '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/dataset/ModelNet40/m40_5_2_12_train_octree.tfrecords'  # The data location
_C.DATA.train.shuffle = 1000  # The shuffle size
_C.DATA.train.n_samples = -1  # Use at most `n_samples` elements from this dataset
_C.DATA.train.batch_size = 32  # Training data batch size
_C.DATA.train.mask_ratio = 0.0  # Mask out some point features

_C.DATA.train.return_iterator = False  # Return the data iterator
_C.DATA.train.return_pts = False  # Also return points

_C.DATA.test = _C.DATA.train.clone()
_C.DATA.test.location = '/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script/dataset/ModelNet40/m40_5_2_12_test_octree.tfrecords'

# MODEL related parameters
_C.MODEL = CN()
_C.MODEL.gpu = (0,)  # The gpu ids
_C.MODEL.logdir = 'logs'  # Directory where to write event logs
_C.MODEL.ckpt = ''  # Restore weights from checkpoint file
_C.MODEL.run = 'train'  # Choose from train or test
_C.MODEL.max_iter = 160000  # Maximum training iterations
_C.MODEL.test_iter = 100  # Test steps in testing phase
_C.MODEL.test_every_iter = 1000  # Test model every n training steps
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

# backup the commands
_C.SYS = CN()
_C.SYS.cmds = ''  # Used to backup the commands

FLAGS = _C
