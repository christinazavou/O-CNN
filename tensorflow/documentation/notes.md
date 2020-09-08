
in TF implementation:
** probably they use two graphs, one with train and one with test because:
1. in train they want to support multi gpu thus reuse=True is useful
2. using TFRecordDataset its easy to make one dataset reading training data and one reading input data and then just pass them as inputs .. instead of feeding training data and then switching dataset to read test...

in OCNN paper they do:
shape classification, shape retrieval, part segmentation
in AOCNN paper they do:
shape classification, autoencoding, shape prediction from an image, shape completion from incomplete point cloud
in MIDNet paper they do:
shape classification, two semantic shape segmentation, shape registration
