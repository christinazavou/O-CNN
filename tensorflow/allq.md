Hello again,

I'm trying to run the Tensorflow implementation for the autoencoder and I have a few questions (excuse me it got too long but i tried to be detailed).

In the repo you have a config file (ae_resnet.yaml) for running a resnet with ".points" tfrecords. I have run this and using the decode_shape gives me pretty good results! However, this is not the case when I try to run an autoencoder with ocnn and ".octree" tfrecords. Below I specify the steps I followed:

1. i used ```python data/completion.py --run generate_dataset``` which downloaded the ".points" files of the completion dataset and generated:
```
shape.ply
shape.points
test.scans.ply
test.scans.points
completion_test_points.tfrecords
completion_test_scans_points.tfrecords
completion_train_points.tfrecords
completion_train_points.camera_path.dict
filelist_test.txt
filelist_test_scans.txt
filelist_train.txt
```
 
2. under each category folder in shape.points i generated a list.txt that includes the paths of its points and run ```octree --filenames category/list.txt --output_path shape.octrees/category --depth 6 --split_label 1 --rot_num 6```

Here, I used --depth 6 and --split_label 1 because I saw these parameters in ae_resnet.yaml.

3. similarly to filelist_test.txt and filelist_train.txt I generated files that include paths of the corresponding octrees (for each .points file path i included the 6 .octree file paths) and then I made octree tfrecords. Specifically these corresponding files:
```
completion_test_octrees.tfrecords
completion_test_scans_octrees.tfrecords
completion_train_octrees.tfrecords
```

4. i run the autoencoder with ae_ocnn.yaml:
```
SOLVER:
  gpu: 0,
  logdir: /output/ocnn_completion/ae/ocnn_b16
  run: train
  max_iter: 20000
  test_iter: 336
  test_every_iter: 400
  step_size: (80000,)
  ckpt_num: 20

DATA:
  train:
    dtype: octree
    depth: 6
    location: /completion_train_octrees.tfrecords
    batch_size: 16
    distort: False
    offset: 0.0
    node_dis: True
    split_label: True

  test: 
    dtype: octree
    depth: 6
    location: /completion_test_octrees.tfrecords
    batch_size: 16
    distort: False
    offset: 0.0
    node_dis: True
    split_label: True

MODEL:
  name: ocnn
  channel: 3
  nout: 32
  depth: 6

LOSS:
  weight_decay: 0.0005
```

Q1. What I noticed is that the resnet with points is taking more time to do the same amount of batch iterations (1 hour equals to 2800 iterations in resnet and 12000 iterations in ocnn) and the accuracy and loss of resnet are much smoother than the ones in ocnn. e.g.
![image](https://user-images.githubusercontent.com/15656466/90230817-cbf05100-de22-11ea-9852-32a255e66f74.png) vs. ![image](https://user-images.githubusercontent.com/15656466/90230846-d6124f80-de22-11ea-9d79-c563295561ca.png)
Also, the decoded octrees from resnet are much better than the ones from ocnn: e.g.
![image](https://user-images.githubusercontent.com/15656466/90230991-1e317200-de23-11ea-9d22-2f5bd2318a4a.png) at 6K batch iterations of resnet vs. ![image](https://user-images.githubusercontent.com/15656466/90231021-2a1d3400-de23-11ea-8cd3-af85358dd625.png) at 20K batch iterations of ocnn.

Q2. I also have a question regarding the input signal. In classification both cls_octree.yaml and cls_points.yaml have input channel 3, which as I understand it represents the normals at the leaf octants, i.e. nx,ny,nz. In the autoencoder however,  ae_resnet.yaml specifies input channel as 4, but using my ae_ocnn.yaml I get error if I don't set input channel to 3. This made me think that I have to generate adaptive octrees, so I repeated step 2. with "--adaptive 4". However, running the ocnn autoencoder showed again the same error i.e. ```F octree_property_op.cc:101] Check failed: channel_ == channel (4 vs. 3)The specified channel_ is wrong.``` I'm also wondering why the cls_points.yaml has model channel 3.


AOCNN paper: from fig4 i see the encoder having skip_connections ... are these the ones referred in the code (where there is flag to use skip_connections)  or not !?

also wondering what does the output of check_octree means:

example of an octree generated with "octree ... --depth 6 --split_label 1 --rot_num 6":
===============
../ocnn_completion/.../02691156/1a04e3eab45ca15dd86060f189eb133_6_2_000.octree infomation:
This is a valid octree!
magic_str:_OCTREE_1.0_
batch_size: 1
depth: 6
full_layer: 2
adaptive_layer: 4
threshold_distance: 2
threshold_normal: 0.1
is_adaptive: 0
has_displace: 0
nnum: 1 8 64 224 1000 4336 17320 0 0 0 0 0 0 0 0 0 
nnum_cum: 0 1 9 73 297 1297 5633 22953 22953 0 0 0 0 0 0 0 
nnum_nempty: 1 8 28 125 542 2165 8177 0 0 0 0 0 0 0 0 0 
total_nnum: 22953
total_nnum_capacity: 22953
channel: 1 1 0 3 0 1 
locations: -1 -1 0 6 0 -1 
bbox_max: 128.635 130.831 120.122 
bbox_min: 17.2621 19.4578 8.74912 
key2xyz: 0
sizeof_octree: 483992
===============

example of an octree generated with "octree ... --depth 6 --split_label 1 --rot_num 6 --adaptive true":
===============
.../ocnn_completion/.../02691156/10aa040f470500c6a66ef8df4909ded9_6_2_000.octree infomation:
This is a valid octree!
magic_str:_OCTREE_1.0_
batch_size: 1
depth: 6
full_layer: 2
adaptive_layer: 4
threshold_distance: 2
threshold_normal: 0.1
is_adaptive: 1
has_displace: 0
nnum: 1 8 64 192 528 744 1200 0 0 0 0 0 0 0 0 0 
nnum_cum: 0 1 9 73 265 793 1537 2737 2737 0 0 0 0 0 0 0 
nnum_nempty: 1 8 24 66 93 150 720 0 0 0 0 0 0 0 0 0 
total_nnum: 2737
total_nnum_capacity: 2737
channel: 1 1 0 3 0 1 
locations: -1 -1 0 -1 0 -1 
bbox_max: 112.246 113.762 118.937 
bbox_min: 3.37262 4.88874 10.0632 
key2xyz: 0
sizeof_octree: 66404
===============

1. i guess adaptive_layer:4 is dummy in the first example because of is_adaptive: 0
2. what is the channel parameter and the locations parameter showing?

I'm also trying to understand the octree_property function, and I have the following:

```
import sys

sys.path.append("../..")
from src.data_parsing import *
from libs import *


class DatasetDebug:

    @staticmethod
    def check_properties():

        filename = 'resources/ModelNetOnly4Samples3/m40_train_points.tfrecords'
        octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                      TransformPoints(distort=False, depth=5, offset=0.55, axis='z', scale=0.0,
                                                      jitter=0.0, angle=[180, 180, 180],
                                                      bounding_sphere=bounding_sphere),
                                      Points2Octree(depth=5))
        octree, label = octrees(filename, 32, shuffle_size=False, return_iterator=False, take=10)

        with tf.Session() as sess:
            # it only lets me use channel=0
            for d in range(-10, 10):
                res = sess.run(octree_property(octree, property_name="split", dtype=tf.float32, depth=d, channel=0))
                print("d ", d, " split ", res.shape)
                res = sess.run(octree_property(octree, property_name="label", depth=d, channel=0, dtype=tf.float32))
                print("d ", d, " label ", res.shape)

            # it only lets me use channel=3
            for d in range(1, 6):
                res = sess.run(octree_property(octree, property_name="feature", depth=d, channel=3, dtype=tf.float32))
                print("d ", d, " feature ", res.shape)

            res = sess.run(octree_property(octree, property_name="feature", depth=-6, channel=3, dtype=tf.float32))
            print("d ", -6, " feature ", res.shape)

            # it only lets me use channel=1
            for d in range(-1, 8):
                res = sess.run(octree_property(octree, property_name="index", depth=d, channel=1, dtype=tf.int32))
                print("d ", d, " index ", res.shape)

            # it only lets me use channel=1
            for d in range(-2, 8):
                res = sess.run(octree_property(octree, property_name="xyz", depth=4, channel=1, dtype=tf.uint32))
                print("d ", d, " xyz ", res.shape)

DatasetDebug.check_properties()
```

and the result i get is:
```
d  -10  split  (0, 1596993073)
d  -10  label  (0, 1596993073)
d  -9  split  (0, 0)
d  -9  label  (0, 0)
d  -8  split  (0, 32)
d  -8  label  (0, 32)
d  -7  split  (0, 5)
d  -7  label  (0, 5)
d  -6  split  (0, 2)
d  -6  label  (0, 2)
d  -5  split  (0, 4)
d  -5  label  (0, 4)
d  -4  split  (0, 0)
d  -4  label  (0, 0)
d  -3  split  (0, 1073741824)
d  -3  label  (0, 1073741824)
d  -2  split  (0, 1036831949)
d  -2  label  (0, 1036831949)
d  -1  split  (0, 0)
d  -1  label  (0, 0)
d  0  split  (0, 32)
d  0  label  (0, 32)
d  1  split  (0, 256)
d  1  label  (0, 256)
d  2  split  (0, 2048)
d  2  label  (0, 2048)
d  3  split  (0, 5872)
d  3  label  (0, 6176)
d  4  split  (0, 23600)
d  4  label  (0, 22944)
d  5  split  (0, 107952)
d  5  label  (0, 100984)
d  6  split  (0, 0)
d  6  label  (0, 0)
d  7  split  (0, 0)
d  7  label  (0, 0)
d  8  split  (0, 0)
d  8  label  (0, 0)
d  9  split  (0, 0)
d  9  label  (0, 0)
d  1  feature  (3, 256)
d  2  feature  (3, 2048)
d  3  feature  (3, 5904)
d  4  feature  (3, 23536)
d  5  feature  (3, 105568)
d  -6  feature  (3, 2)
d  -1  index  (1, 0)
d  0  index  (1, 32)
d  1  index  (1, 256)
d  2  index  (1, 2048)
d  3  index  (1, 6160)
d  4  index  (1, 23040)
d  5  index  (1, 105248)
d  6  index  (1, 0)
d  7  index  (1, 0)
d  -2  xyz  (1, 23600)
d  -1  xyz  (1, 22944)
d  0  xyz  (1, 23664)
d  1  xyz  (1, 22976)
d  2  xyz  (1, 24048)
d  3  xyz  (1, 23040)
d  4  xyz  (1, 23536)
d  5  xyz  (1, 23296)
d  6  xyz  (1, 23328)
d  7  xyz  (1, 23600)
```

can i get "shuffle key" from octree_property?
is "label" in "octree_property" giving the "which indicates that it is the p-th non-empty octant in the sorted octant list of the l-th depth." ?
can i get "input signal" from octree_property? is it the "xyz" property or the "feature" property ?
can i get "cnn features" from octree_property? if i apply a convolution and call again octree_property with this i will get a new result in each training step?

displace = translate the location of the object ? that's why in input/output of decoder we have different locations ?

the octree2mesh ... if i have a predicted octree from OCNN i will see equal size patches and if i predicted octree from AOCNN i will see different-size patches?!

what is the 4-dim input signal ? why autoencoder with points wants channel 4 and with octrees wants channel 3 ?

the autoencoder results are both input and output saved as ".octree" even if we use resnet or ocnn ... why/how

However, I'm not sure I understand the difference of the code using 'points' and using 'octree'. In the classification code, using 'octree' calls the DatasetFactory that reads from octree tfrecords, and using 'points' calls the DatasetFactory that reads from point tfrecords , transforms them and merges them to octrees. So is 'merge octree' a function that can be run either on ".octree" or on ".points" and it generates either merged ".octrees" or merged ".points"? Because both formats are in bytes and i can't see their difference; Also, can I call "octree_property" function both on data loaded from ".points" file or ".octree" file but with different arguments? Which arguments can be used for each one?