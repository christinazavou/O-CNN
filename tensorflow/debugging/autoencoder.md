 encoder:
'data_d6' = "ocnn_encoder_1/signal_gt/OctreeProperty:0", shape=(4, ?) ===> (4,212216)
'data_front' = "ocnn_encoder_1/front/conv_bn_relu/Relu:0", shape=(1, 32, ?, 1) ===> (1,32,212216)
'before code' = "ocnn_encoder_1/down_3/conv_bn_relu/Relu:0", shape=(1, 256, ?, 1) ===> (1,256,1024,1)
'after code' = "ocnn_encoder_1/down_3/conv_bn_relu/Relu:0", shape=(1, 256, ?, 1) ===> (1,256,1024,1)

decoder:
'data_d2': 'ocnn_decoder/resblock_2_2/Relu:0' shape=(1, 256, ?, 1) ===> (1,256,1024,1)
'logit_d2':'ocnn_decoder/predict_2/transpose:0' shape=(?, 2) ===> (1024,2)
'label_gt_d2': 'ocnn_decoder/loss_2/label_gt/OctreeProperty:0' shape=(1, ?) ===> (1,1024)
'up_data_d2': 'ocnn_decoder/up_2/deconv_bn_relu/Relu:0' shape=(1, 64, ?, 1) ===> (1,64,3080,1) 
'data_d3': 'ocnn_decoder/resblock_3_2/Relu:0' shape=(1, 256, ?, 1) ===> (1,256,3080,1)
'logit_d3': 'ocnn_decoder/predict_3/transpose:0' shape=(?, 2) ===> (3080,2)
'label_gt_d3': 'ocnn_decoder/loss_3/label_gt/OctreeProperty:0' shape=(1, ?) ===>(1,3080)
'up_data_d3': 'ocnn_decoder/up_3/deconv_bn_relu/Relu:0' shape=(1, 256, ?, 1) ===> (1,256,12256,1)
'data_d4': 'ocnn_decoder/resblock_4_2/Relu:0' shape=(1, 128, ?, 1) ===> (1,128,12256,1)
'logit_d4': 'ocnn_decoder/predict_4/transpose:0' shape=(?, 2) ===> (12256,2)
'label_gt_d4': 'ocnn_decod... ===> (1,12256)
'up_data_d4' = "ocnn_decoder/up_4/deconv_bn_relu/Relu:0", shape=(1, 256, ?, 1) ===> (1,256,50048,1)
'logit_d5' = "ocnn_decoder/predict_5/transpose:0", shape=(?, 2) ===> (50048,2)
'data_d5' = "ocnn_decoder/resblock_5_2/Relu:0", shape=(1, 64, ?, 1)
'up_data_d5' = "ocnn_decoder/up_5/deconv_bn_relu/Relu:0", shape=(1, 128, ?, 1) ===> (1,128,212216,1)
'label_gt_d5' = "ocnn_decoder/loss_5/label_gt/OctreeProperty:0", shape=(1, ?) ===> (1,50048)
'data_d6' = "ocnn_decoder/resblock_6_2/Relu:0", shape=(1, 32, ?, 1), dtype=float32) ===> (1,32,212216,1)
'logit_d6' = "ocnn_decoder/predict_6/transpose:0", shape=(?, 2), dtype=float32) ===> (212216,2)
'signal' = "ocnn_decoder/regress_6/Tanh:0", shape=(1, 4, ?, 1), dtype=float32) ===> (1,4,212216,1)
'signal_gt' = "ocnn_decoder/loss_regress/signal_gt/OctreeProperty:0", shape=(4, ?) ===> (4,212216)
'label_gt_d6' = "ocnn_decoder/loss_6/label_gt/OctreeProperty:0", shape=(1, ?)  ===> (1,212216)


in resources: completion_test_octrees.tfrecords has 48 records
              completion_test_points.tfrecords has 4 records