
in TF implementation:
** probably they use two graphs, one with train and one with test because:
1. in train they want to support multi gpu thus reuse=True is useful
2. using TFRecordDataset its easy to make one dataset reading training data and one reading input data and then just pass them as inputs .. instead of feeding training data and then switching dataset to read test...
