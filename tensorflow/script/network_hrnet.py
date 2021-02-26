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
    debug_checks = {}  # TODO remove if statement
    # if depth > 5: block_num = block_num // 2  # !!! whether should we add this !!!
    for i in range(block_num):
        with tf.variable_scope('resblock_d%d_%d' % (depth, i)):
            bottleneck = 4 if channel < 256 else 8  # TODO change bottleneck size dynamically
            data = octree_resblock(data, octree, depth, channel, 1, training, bottleneck)
            debug_checks['{}/data'.format(tf.get_variable_scope().name)] = data
    return data, debug_checks


def branch_channels(channel, i):
    return (2 ** i) * channel


def branches(data, octree, depth, channel, block_num, training):
    for i in range(len(data)):
        with tf.variable_scope('branch_%d' % (depth - i)):
            depth_i, channel_i = depth - i, branch_channels(channel, i)
            # if channel_i > 256: channel_i = 256
            data[i], dc = branch(data[i], octree, depth_i, channel_i, block_num, training)
    return data


def trans_func(data_in, octree, d0, d1, training, upsample):
    data = data_in
    channel0 = int(data.shape[1])
    channel1 = channel0 * (2 ** (d0 - d1))
    # if channel1 > 256: channel1 = 256  ## !!! clip the channel to 256
    # no relu for the last feature map
    with tf.variable_scope('trans_%d_%d' % (d0, d1)):
        if d0 > d1:  # downsample, transitioning to smaller depth
            for d in range(d0, d1, -1):
                with tf.variable_scope('down_%d' % d):  # transfer features from depth d to d1
                    data, _ = octree_max_pool(data, octree, d)
            with tf.variable_scope('conv1x1_%d' % (d1)):
                data = octree_conv1x1_bn(data, channel1, training)  # get channels to wanted size
        elif d0 < d1:  # upsample, transitioning to bigger depth
            for d in range(d0, d1, 1):
                with tf.variable_scope('up_%d' % d):
                    if d == d0:
                        data = octree_conv1x1_bn(data, channel1, training)
                    data = OctreeUpsample(upsample)(data, octree, d)
        else:  # do nothing, return data_in without any changes
            pass
    return data


def transitions(data, octree, depth, training, upsample='neareast'):
    num = len(data)
    features = [[0] * num for _ in range(num + 1)]
    for i in range(num):
        for j in range(num + 1):
            d0, d1 = depth - i, depth - j
            features[j][i] = trans_func(data[i], octree, d0, d1, training, upsample)

    outputs = [None] * (num + 1)
    for j in range(num + 1):
        with tf.variable_scope('fuse_%d' % (depth - j)):
            outputs[j] = tf.nn.relu(tf.add_n(features[j]))
    return outputs


def front_layer_channeld(channel, d, d1):
    return channel / 2 ** (d - d1 + 1)


class HRNet:
    def __init__(self, flags):
        self.tensors = dict()
        self.flags = flags

    def network_seg(self, octree, training, reuse=False, pts=None, mask=None):
        debug_checks = {}
        with tf.variable_scope('ocnn_hrnet', reuse=reuse):
            ## backbone
            convs, dc = self.backbone(octree, training)
            debug_checks.update(dc)
            self.tensors['convs'] = convs

            ## header
            with tf.variable_scope('seg_header'):
                if pts is None:
                    logit, dc = self.seg_header(convs, octree, self.flags.nout, mask, training)
                else:
                    logit, dc = self.seg_header_pts(convs, octree, self.flags.nout, pts, training)
                    debug_checks.update(dc)
                self.tensors['logit_seg'] = logit
        return logit, debug_checks

    def seg_header(self, inputs, octree, nout, mask, training):
        debug_checks = {}
        feature = self.points_feat(inputs, octree)  # d5-128,d4-256,d3-516 = 896 feats

        factor = self.flags.factor
        if self.flags.with_d0:
            feature = OctreeUpsample('linear')(feature, octree, self.flags.depth - 1, mask)
            debug_checks['{}/feature(linear_ups)'.format(tf.get_variable_scope().name)] = feature
            convd0 = self.tensors['front/convd0']  # (1, C, H, 1)
            if mask is not None:
                convd0 = tf.boolean_mask(convd0, mask, axis=2)
            feature = tf.concat([feature, convd0], axis=1)  # append input depth features, d6-32 =>928 feats
            debug_checks['{}/convd0'.format(tf.get_variable_scope().name)] = convd0
            debug_checks['{}/feature(concat)'.format(tf.get_variable_scope().name)] = feature
        else:
            if mask is not None:
                feature = tf.boolean_mask(feature, mask, axis=2)

        with tf.variable_scope('predict_%d_with%s_convd0' % (self.flags.depth, "" if self.flags.with_d0 else "out")):
            logit = predict_module(feature, nout, 128 * factor, training)  # 2-FC
            logit = tf.transpose(tf.squeeze(logit, [0, 3]))  # (1, C, H, 1) -> (H, C)

        return logit, debug_checks

    def seg_header_pts(self, inputs, octree, nout, pts, training):
        debug_checks = {}
        feature = self.points_feat(inputs, octree)  # The resolution is 5-depth
        # d5-128,d4-256,d3-516 = 896 feats

        depth, factor = self.flags.depth, self.flags.factor
        xyz, ids = tf.split(pts, [3, 1], axis=1)  # get xyz and octree id in current batch
        xyz = xyz + 1.0  # [0, 2]
        ptsd1 = tf.concat([xyz * (2.0 ** (depth - 2)), ids], axis=1)  # [0, 32], d6 resolution
        debug_checks["{}pts/ptsd1".format(tf.get_variable_scope().name)] = ptsd1
        feature = octree_bilinear_v3(ptsd1, feature, octree,
                                     depth=depth - 1)  # transfer octree features to pts
        debug_checks["{}pts/feature(bilinear)".format(tf.get_variable_scope().name)] = feature
        if self.flags.with_d0:
            convd0 = self.tensors['front/convd0']  # The resolution is 6-depth
            ptsd0 = tf.concat([xyz * (2 ** (depth - 1)), ids], axis=1)  # [0, 64]
            debug_checks["{}pts/ptsd0".format(tf.get_variable_scope().name)] = ptsd0
            convd0 = octree_nearest_interp(ptsd0, convd0, octree, depth=depth)
            debug_checks["{}pts/convd0(nearinterp)".format(tf.get_variable_scope().name)] = convd0
            feature = tf.concat([feature, convd0], axis=1)
            debug_checks["{}pts/feature(concat)".format(tf.get_variable_scope().name)] = feature

        with tf.variable_scope('predict_%d_with%s_convd0' % (self.flags.depth, "" if self.flags.with_d0 else "out")):
            logit = predict_module(feature, nout, 128 * factor, training)  # 2-FC
            logit = tf.transpose(tf.squeeze(logit, [0, 3]))  # (1, C, H, 1) -> (H, C)

        return logit, debug_checks

    def points_feat(self, inputs, octree):
        data = [t for t in inputs]
        depth, factor, num = self.flags.depth - 1, self.flags.factor, len(inputs)
        assert (self.flags.depth >= depth)
        for i in range(1, num):
            with tf.variable_scope('up_%d' % i):
                for j in range(i):
                    d = depth - i + j
                    data[i] = OctreeUpsample(self.flags.upsample)(data[i], octree, d)
        feature = tf.concat(data, axis=1)  # the resolution is depth-5
        return feature

    def cls_header(self, inputs, octree, nout, training):
        data = [t for t in inputs]
        channel = [int(t.shape[1]) for t in inputs]
        depth, factor, num = self.flags.depth, self.flags.factor, len(inputs)
        assert (self.flags.depth >= depth)
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
        depth = flags.depth
        with tf.variable_scope('signal'):
            data = octree_property(octree, property_name='feature', dtype=tf.float32,
                                   depth=depth, channel=flags.channel)
            data = tf.reshape(data, [1, flags.channel, -1, 1])  # [1,channels,no. octants,1]
            debug_checks['{}/data(feature)'.format(tf.get_variable_scope().name)] = data
            if flags.signal_abs: data = tf.abs(data)

        # front
        convs = [None]
        channel, d1 = 64 * flags.factor, depth - 1  # chosen resolution, main working depth (depth-1)
        convs[0] = self.front_layer(data, octree, depth, d1, channel, training)

        # stages, how many depths to consider in HRNet architecture
        stage_num = flags.stages
        for stage in range(1, stage_num + 1):
            with tf.variable_scope('stage_%d' % stage):
                convs = branches(convs, octree, d1, channel, flags.resblock_num, training)
                if stage == stage_num: break
                # move to shallower depth
                convs = transitions(convs, octree, depth=d1, training=training, upsample=flags.upsample)
        return convs, debug_checks

    def front_layer(self, data, octree, d0, d1, channel, training):
        conv = data
        with tf.variable_scope('front'):
            for d in range(d0, d1, -1):
                with tf.variable_scope('depth_%d' % d):
                    channeld = front_layer_channeld(channel, d, d1)
                    conv = octree_conv_bn_relu(conv, octree, d, channeld, training)
                    self.tensors['front/convd0'] = conv  # TODO: add a resblock here?
                    conv, _ = octree_max_pool(conv, octree, d)
            with tf.variable_scope('depth_%d' % d1):
                conv = octree_conv_bn_relu(conv, octree, d1, channel, training)
                self.tensors['front/convd1'] = conv
        return conv
