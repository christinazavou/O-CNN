
- in the code he is passing a lot of functions around as arguments, and the functions he is passing is many times the `__call__` of a class !!!

 
### run_cls.py:

first call graph with train and reuse=false:
with tf.variable_scope("ocnn", reuse=False)
    
    variable data:
        transform shape into [1, 3(channels), -1, 1] where i guess -1 is for the depth ? or for the length of arrays containing octant info?
        
    loop for depths [max_depth,...,2]

### Questions
- is 'test' scope because in 'train' scope he does anyway a split of train+validation ?

- octree_conv_memory and octree_conv_fast ... i dont understand them 

- sto build_train_graph giati train_tensors reuse = false kai test_tensors reuse = true ? na do ta functions apo kato tous ti xrisimopoioun..

- to summaries pos to kanei ?

- sto test_octree2col:
    - to data_in pou exei shape 1,3,8,1 einai diladi 1 sample, 3 channels, 8 features tou octree(octant) ? 
    - to idx_maps ti einai ?
    - to vi ti einai ?
    - to kernel_size pou einai 3 kathe fora diladi [num_filters, wifht, height] or something else?
    - pos douleuei to octree2col ? examples:
        - input (1,3,8,1), kernel [3,3,3], stride 1, output (3,27,8)
        - input (1,3,8,1), kernel [2,2,2], stride 1, output (3,8,8)
        - input (1,3,8,1), kernel [3,1,1], stride 1, output (3,3,8)
        - input (1,3,8,1), kernel [3,3,1], stride 1, output (3,9,8)
        - input (1,3,8,1), kernel [3,3,3], stride 2, output (3,27,1)
        - input (1,3,8,1), kernel [2,2,2], stride 2, output (3,8,1)

- giati data shape [1, 3 (channels), 152, 1] ?? giati using only height ? 


 - test_octree_deconv will raise exception ..


- what is octree_search doing ? 
- what is octree_property(property_name='index') and octree_property(property_name='xyz') doing?


- octree_batch(octree_samples(['octree_1', 'octree_2'])) this probably concatenates the two different octrees into one super-octree as explained in paper

- what is the 'split_label' ?

- ~~is there a run mode where both train graph and test graph are created?~~ yes in 'train' mode


** probably they use two graphs, one with train and one with test because:
1. in train they want to support multi gpu thus reuse=True is useful
2. using TFRecordDataset its easy to make one dataset reading training data and one reading input data and then just pass them as inputs .. instead of feeding training data and then switching dataset to read test...


