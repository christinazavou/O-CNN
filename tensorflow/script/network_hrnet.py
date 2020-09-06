import tensorflow as tf
from ocnn import *

class OctreeUpsample:
  def __init__(self, upsample='nearest'):
    self.upsample = upsample
  
  def __call__(self, data, octree, d, mask=None):
    if self.upsample == 'nearest':
      data = octree_tile(data, octree, d)
    else:
      data = octree_bilinear(data, octree, d, d + 1, mask)
    return data


def branch(data, octree, depth, channel, block_num, training):
  debug_checks = {}
  if depth > 5: block_num = block_num // 2 # !!! whether should we add this !!!
  print("branch with depth ={}, channel = {}, (res)block_num = {}".format(depth, channel, block_num))
  for i in range(block_num):
    with tf.variable_scope('resblock_d%d_%d' % (depth, i)):
      # data = octree_resblock2(data, octree, depth, channel, training)
      bottleneck = 4 if channel < 256 else 8
      data = octree_resblock(data, octree, depth, channel, 1, training, bottleneck)
      debug_checks['{}/data'.format(tf.get_variable_scope().name)] = data
  return data, debug_checks

def branch_channels(channel, i):
  return (2 ** i) * channel

def branches(data, octree, depth, channel, block_num, training):
  debug_checks = {}
  print("branches at depth {} with len(data)={}".format(depth, len(data)))
  for i in range(len(data)):
    with tf.variable_scope('branch_%d' %  (depth - i)):
      depth_i, channel_i = depth - i, branch_channels(channel, i)
      # if channel_i > 256: channel_i = 256
      data[i], dc = branch(data[i], octree, depth_i, channel_i, block_num, training)
      debug_checks.update(dc)
  return data, debug_checks

def trans_func(data_in, octree, d0, d1, training):
  data = data_in
  channel0 = int(data.shape[1])
  channel1 = channel0 * (2 ** (d0 - d1))
  # if channel1 > 256: channel1 = 256  ## !!! clip the channel to 256
  # no relu for the last feature map
  with tf.variable_scope('trans_%d_%d' % (d0, d1)):
    if d0 > d1:   # downsample
      for d in range(d0, d1 + 1, -1):
        with tf.variable_scope('down_%d' % d):
          data = octree_conv_bn_relu(data, octree, d, channel0/4, training, stride=2)
      with tf.variable_scope('down_%d' % (d1 + 1)):
        data = octree_conv_bn(data, octree, d1 + 1, channel1, training, stride=2)
    elif d0 < d1: # upsample
      for d in range(d0, d1, 1): 
        with tf.variable_scope('up_%d' % d):
          if d == d0:
            data = octree_conv1x1_bn(data, channel1, training)
          data = octree_tile(data, octree, d)
    else:        # do nothing
      pass
  return data

def trans_func(data_in, octree, d0, d1, training, upsample):
  data = data_in
  channel0 = int(data.shape[1])
  channel1 = channel0 * (2 ** (d0 - d1))
  # if channel1 > 256: channel1 = 256  ## !!! clip the channel to 256
  # no relu for the last feature map
  with tf.variable_scope('trans_%d_%d' % (d0, d1)):
    if d0 > d1:   # downsample
      for d in range(d0, d1, -1):
        with tf.variable_scope('down_%d' % d):
          data, _ = octree_max_pool(data, octree, d)
      with tf.variable_scope('conv1x1_%d' % (d1)):
        data = octree_conv1x1_bn(data, channel1, training)
    elif d0 < d1: # upsample
      for d in range(d0, d1, 1): 
        with tf.variable_scope('up_%d' % d):
          if d == d0:
            data = octree_conv1x1_bn(data, channel1, training)
          data = OctreeUpsample(upsample)(data, octree, d)
    else:        # do nothing
      pass
  return data

def transitions(data, octree, depth, training, upsample='neareast'):
  debug_checks = {}
  num = len(data)
  features = [[0]*num for i in range(num + 1)]
  for i in range(num):
    for j in range(num + 1):
      d0, d1 = depth - i, depth - j
      print("i {}, j {}, d0 {}, d1 {}".format(i, j, d0, d1))
      features[j][i] = trans_func(data[i], octree, d0, d1, training, upsample)
  debug_checks['{}/transitions_features'.format(tf.get_variable_scope().name)] = features

  outputs = [None] *(num + 1)
  for j in range(num + 1):
    with tf.variable_scope('fuse_%d' % (depth - j)):
      outputs[j] = tf.nn.relu(tf.add_n(features[j]))
  debug_checks['{}'.format(tf.get_variable_scope().name)] = outputs
  return outputs, debug_checks


def front_layer_channeld(channel, d, d1):
  return channel / 2 ** (d - d1 + 1)


class HRNet:
  def __init__(self, flags):
    self.tensors = dict()
    self.flags = flags

  def network(self, octree, training, mask=None, reuse=False):
    flags = self.flags
    with tf.variable_scope('ocnn_hrnet', reuse=reuse):
      # backbone
      convs, db = self.backbone(octree, training)
      self.tensors['convs'] = convs

      # header
      nout_cls, nout_seg = flags.nouts[0], flags.nouts[1]
      with tf.variable_scope('seg_header'):
        logit_seg = self.seg_header(convs, octree, nout_seg, mask, training)
        self.tensors['logit_seg'] = logit_seg

      with tf.variable_scope('cls_header'):
        logit_cls = self.cls_header(convs, octree, nout_cls, training)
        self.tensors['logit_cls'] = logit_cls
    return self.tensors

  def network_cls(self, octree, training, reuse=False):
    with tf.variable_scope('ocnn_hrnet', reuse=reuse):
      # backbone
      convs, db = self.backbone(octree, training)
      self.tensors['convs'] = convs

      # header
      with tf.variable_scope('cls_header'):
        logit = self.cls_header(convs, octree, self.flags.nout, training)
        self.tensors['logit_cls'] = logit
    return logit

  def network_seg(self, octree, training, reuse=False, pts=None, mask=None):
    debug_checks = {}
    with tf.variable_scope('ocnn_hrnet', reuse=reuse):
      ## backbone
      convs, dc = self.backbone(octree, training)
      debug_checks.update(dc)
      debug_checks.update({'backbone': convs})
      self.tensors['convs'] = convs

      ## header
      with tf.variable_scope('seg_header'): 
        if pts is None:
          logit = self.seg_header(convs, octree, self.flags.nout, mask, training)
        else:
          logit, dc = self.seg_header_pts(convs, octree, self.flags.nout, pts, training)
          debug_checks.update(dc)
        self.tensors['logit_seg'] = logit
    return logit, debug_checks

  def seg_header(self, inputs, octree, nout, mask, training):
    debug_checks = {}
    feature, db = self.points_feat(inputs, octree)
    debug_checks.update(db)

    depth_out, factor = self.flags.depth_out, self.flags.factor
    if depth_out == 6:
      feature = OctreeUpsample('linear')(feature, octree, 5, mask)
      conv6 = self.tensors['front/conv6']  # (1, C, H, 1)
      if mask is not None:
        conv6 = tf.boolean_mask(conv6, mask, axis=2)
      feature = tf.concat([feature, conv6], axis=1)
    else:
      if mask is not None:
        feature = tf.boolean_mask(feature, mask, axis=2)

    # feature = octree_conv1x1_bn_relu(feature, 1024, training=training)
    with tf.variable_scope('predict_%d' % depth_out):
      logit = predict_module(feature, nout, 128 * factor, training) # 2-FC
      logit = tf.transpose(tf.squeeze(logit, [0, 3])) # (1, C, H, 1) -> (H, C)  
    return logit

  def seg_header_pts(self, inputs, octree, nout, pts, training):
    debug_checks = {}
    feature, dc = self.points_feat(inputs, octree)  # The resolution is 5-depth
    debug_checks.update(dc)
    
    depth_out, factor = self.flags.depth_out, self.flags.factor
    xyz, ids = tf.split(pts, [3, 1], axis=1)
    debug_checks['{}pts/xyz'.format(tf.get_variable_scope().name)] = xyz
    debug_checks['{}pts/ids'.format(tf.get_variable_scope().name)] = ids
    xyz = xyz + 1.0                                             # [0, 2]
    pts5 = tf.concat([xyz * 16.0, ids], axis=1)                 # [0, 32]
    debug_checks["{}pts/pts5".format(tf.get_variable_scope().name)] = pts5
    feature = octree_bilinear_v3(pts5, feature, octree, depth=5)
    debug_checks["{}pts/feature(bilinear)".format(tf.get_variable_scope().name)] = feature
    if depth_out == 6:
      conv6 = self.tensors['front/conv6']     # The resolution is 6-depth
      pts6  = tf.concat([xyz * 32.0, ids], axis=1)              # [0, 64]
      debug_checks["{}pts/pts6".format(tf.get_variable_scope().name)] = pts6
      conv6 = octree_nearest_interp(pts6, conv6, octree, depth=6)
      debug_checks["{}pts/conv6(nearinterp)".format(tf.get_variable_scope().name)] = conv6
      feature = tf.concat([feature, conv6], axis=1)
      debug_checks["{}pts/feature(concat)".format(tf.get_variable_scope().name)] = feature

    with tf.variable_scope('predict_%d' % depth_out):
      logit = predict_module(feature, nout, 128 * factor, training) # 2-FC
      debug_checks["{}pts/logit".format(tf.get_variable_scope().name)] = logit
      logit = tf.transpose(tf.squeeze(logit, [0, 3])) # (1, C, H, 1) -> (H, C)
      debug_checks["{}pts/logit_transposed".format(tf.get_variable_scope().name)] = logit
    return logit, debug_checks

  def points_feat(self, inputs, octree):
    debug_checks = {}
    data = [t for t in inputs]
    depth, factor, num = 5, self.flags.factor, len(inputs)
    assert(self.flags.depth >= depth)
    for i in range(1, num):
      with tf.variable_scope('up_%d' % i):
        for j in range(i):
          d = depth - i + j
          print("i {} j {} d {}".format(i, j, d))
          data[i] = OctreeUpsample(self.flags.upsample)(data[i], octree, d)
          debug_checks["{}/upsample_i{}j{}d{}".format(tf.get_variable_scope().name, i, j, d)] = data[i]
    debug_checks["{}/upsampled".format(tf.get_variable_scope().name)] = data
    feature = tf.concat(data, axis=1)  # the resolution is depth-5
    debug_checks["{}/feature(concat)".format(tf.get_variable_scope().name)] = feature
    return feature, debug_checks

  def cls_header(self, inputs, octree, nout, training):
    data = [t for t in inputs]
    channel = [int(t.shape[1]) for t in inputs]
    depth, factor, num = 5, self.flags.factor, len(inputs)
    assert(self.flags.depth >= depth)
    for i in range(num):
      conv = data[i]
      d = depth - i
      with tf.variable_scope('down_%d' % d):
        for j in range(2 - i):
          with tf.variable_scope('down_%d' % (d - j)):
            conv, _ = octree_max_pool(conv, octree, d - j)
        data[i] = conv

    features = tf.concat(data, axis=1)
    # with tf.variable_scope("fc0"):
    #   conv = octree_conv1x1_bn_relu(features, 256, training)
    # with tf.variable_scope("fc1"):
    #   conv = octree_conv1x1_bn_relu(conv, 512 * factor, training)
    with tf.variable_scope("fc1"):
      conv = octree_conv1x1_bn_relu(features, 512 * factor, training)
      
    fc1 = octree_global_pool(conv, octree, depth=3)
    self.tensors['fc1'] = fc1
    if self.flags.dropout[0]:
      fc1 = tf.layers.dropout(fc1, rate=0.5, training=training)

    with tf.variable_scope("fc2"):
      # with tf.variable_scope('fc2_pre'):
      #   fc1 = fc_bn_relu(fc1, 512, training=training) 
      logit = dense(fc1, nout, use_bias=True)    
      self.tensors['fc2'] = logit
    return logit

  def backbone(self, octree, training):
    debug_checks = {}
    flags = self.flags
    depth, channel = flags.depth, 64 * flags.factor
    with tf.variable_scope('signal'):
      data = octree_property(octree, property_name='feature', dtype=tf.float32,
                            depth=depth, channel=flags.channel)
      data = tf.reshape(data, [1, flags.channel, -1, 1])
      debug_checks['{}/data(feature)'.format(tf.get_variable_scope().name)] = data
      if flags.signal_abs: data = tf.abs(data)

    # front
    convs = [None]
    channel, d1 = 64 * flags.factor, 5
    convs[0], dc = self.front_layer(data, octree, depth, d1, channel, training)
    debug_checks.update(dc)

    # stages
    stage_num = 3
    print("we'll use 3 stages")
    for stage in range(1, stage_num + 1):
      with tf.variable_scope('stage_%d' % stage):
        convs, dc = branches(convs, octree, d1, channel, flags.resblock_num, training)
        debug_checks.update(dc)
        if stage == stage_num: break
        convs, dc = transitions(convs, octree, depth=d1, training=training, upsample=flags.upsample)
        debug_checks.update(dc)
    return convs, debug_checks

  def front_layer(self, data, octree, d0, d1, channel, training):
    debug_checks = {}
    conv = data
    with tf.variable_scope('front'):
      print("in front we use depths {} to {}".format(d0, d1))
      for d in range(d0, d1, -1):
        with tf.variable_scope('depth_%d' % d):
          channeld = front_layer_channeld(channel, d, d1)
          print("in depth {} with input {} we will output {} channels".format(d, channel, channeld))
          conv = octree_conv_bn_relu(conv, octree, d, channeld, training)
          debug_checks["{}/conv".format(tf.get_variable_scope().name)] = conv
          self.tensors['front/conv6'] = conv # TODO: add a resblock here?
          conv, _ = octree_max_pool(conv, octree, d)
          debug_checks["{}/conv_pooled".format(tf.get_variable_scope().name)] = conv
      with tf.variable_scope('depth_%d' % d1):
        conv = octree_conv_bn_relu(conv, octree, d1, channel, training)
        self.tensors['front/conv5'] = conv
        debug_checks["{}/front/conv5".format(tf.get_variable_scope().name)] = conv
    return conv, debug_checks
