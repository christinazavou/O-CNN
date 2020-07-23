Useful functions:
```python
import tensorflow as tf
tf.train.piecewise_constant(...)
```

```python
from yacs.config import CfgNode as CN
FLAGS = CN()
...
def _update_config(FLAGS, args):
  FLAGS.defrost()
  if args.config:
    FLAGS.merge_from_file(args.config)
  if args.opts:
    FLAGS.merge_from_list(args.opts)
  FLAGS.SYS.cmds = ' '.join(sys.argv)
  FLAGS.freeze()
```

```python
import shutil
shutil.copy2(args.config, logdir)
```
    
```python
eval('self.{}()'.format(self.flags.run))
```

```python
# __init__ vs __call__

class Foo:
    def __init__(self, a, b, c):
        pass
    def __call__(self, a, b, c):
        pass

x = Foo(1, 2, 3) # calls __init__
x(4, 5, 6) # calls __call__
```

- in the code he is passing a lot of functions around as arguments, and the functions he is passing is many times the `__call__` of a class !!!

```python
tf.variable_scope(scope_name, reuse)
# reuse set to 
# None: means this tf.variable_scope() inherits parent variable scope reuse mode
# True: makes this variable scope and sub scopes to a reuse mode if sub scopes have set reuse = None.
# tf.AUTO_REUSE: it will create variables if they do not exist, and return them otherwise.

# When we use tf.variable_scope(), we should use tf.get_variable() function to create or return an existing variable. If you use tf.Variable(), it will create a new variable no matter what value the reuse parameter is.

# Note: tf.get_variable() can create a new tensorflow only if reuse = None or tf.AUTO_REUSE. if reuse = True, tf.get_variable() can only return an existing variable created by tf.get_variable(); it can not create a new one.

# so basically we can have two python variables that point to the same tensroflow variable!! :
def layer_weight():
    with tf.variable_scope("weight", reuse = tf.AUTO_REUSE):
        w = tf.get_variable(name = 'w',initializer = tf.random_normal(shape=[2,2], mean=0, stddev=1))
    return w
w1 = layer_weight()
w2 = layer_weight()
# w1 is same as w2
```

```python
tf.layers.batch_normalization(conv, axis=1, training=training)
# can be used instead of manually generating moving averages and update ops ...
```

```python
# for testing tensorflow:

class Octree2ColTest(tf.test.TestCase):
    ...
    def test_forward(self):
        with self.cached_session():     
            ...
```
   
- also in tests:
    -   tf.test.compute_gradient(..) to compute theoretical and numerical jacobian gradients !!
    -   self.assertAllClose() to assert that each number of arrays differs from its equivalent no more than a small number !!
    

- run() and eval():
    - op.run() is a shortcut for calling tf.get_default_session().run(op)
    - t.eval() is a shortcut for calling tf.get_default_session().run(t)
    - The difference is in Operations vs. Tensors. Operations use run() and Tensors use eval().
    - basically these are equivalent:
    ```python
    # Using `Session.run()`.
    sess = tf.Session()
    c = tf.constant(5.0)
    print(sess.run(c))
    
    # Using `Tensor.eval()`.
    c = tf.constant(5.0)
    with tf.Session():
      print(c.eval())
    ```

- tensor vs operation:
    - A tensor represents a rectangular array of data.
    - An operation represents a graph node that performs computation on tensors.
    
```python
tf.data.TFRecordDataset('path').take(-1)
# with -1 we take all samples of dataset
```
 
 
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

- some tests are skipped (probably they dont have any assertions):
    - , test_octree_conv, test_octree_deconv, test_octree_gather, test_octree_search
- some tests are ok:
    - test_octree_2col, test_octree_align, test_octree_linear, test_octree_nearest, test_octree_property
- some tests fail: (probably files are not there)
    - test_octree_grow, test_points_property, test_transform_points
    
- giati data shape [1, 3 (channels), 152, 1] ?? giati using only height ? 