Hello again,

I'm trying to run the Tensorflow implementation for the autoencoder and I have a few questions (excuse me it got too long but i tried to be detailed).

In the repo you have a config file (ae_resnet.yaml) for running a resnet with ".points" tfrecords. I have run this and using the decode_shape gives me pretty good results! However, this is not the case when I try to run an autoencoder with ocnn and ".octree" tfrecords. 
 
####Here are the steps I followed:

**Step 1.** i used ```python data/completion.py --run generate_dataset``` which downloaded the ".points" files of the completion dataset and generated:
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
 
**Step 2.** under each category folder in shape.points i generated a list.txt that includes the paths of its points and run ```octree --filenames category/list.txt --output_path shape.octrees/category --depth 6 --split_label 1 --rot_num 6```

Here, I used --depth 6 and --split_label 1 because I saw these parameters in ae_resnet.yaml.

**Step 3.** similarly to filelist_test.txt and filelist_train.txt I generated files that include paths of the corresponding octrees (for each .points file path i included the 6 .octree file paths) and then I made octree tfrecords. Specifically these corresponding files:
```
completion_test_octrees.tfrecords
completion_test_scans_octrees.tfrecords
completion_train_octrees.tfrecords
```

**Step 4.** i run the autoencoder with ae_ocnn.yaml:
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

#### Results I got

What I noticed is that the resnet with points is taking more time to do the same amount of batch iterations (1 hour equals to 2800 iterations in resnet and 12000 iterations in ocnn) and the accuracy and loss of resnet are much smoother than the ones in ocnn. e.g.
![image](https://user-images.githubusercontent.com/15656466/90230817-cbf05100-de22-11ea-9852-32a255e66f74.png) vs. ![image](https://user-images.githubusercontent.com/15656466/90230846-d6124f80-de22-11ea-9d79-c563295561ca.png)

Also, the decoded octrees from resnet are much better than the ones from ocnn: e.g.
![image](https://user-images.githubusercontent.com/15656466/90230991-1e317200-de23-11ea-9d22-2f5bd2318a4a.png) at 6K batch iterations of resnet vs. ![image](https://user-images.githubusercontent.com/15656466/90231021-2a1d3400-de23-11ea-8cd3-af85358dd625.png) at 20K batch iterations of ocnn.

#### Below I list my questions:

**Q1.** I guess my configuration for ocnn autoencoder is wrong and I have a question regarding the input signal. In classification both cls_octree.yaml and cls_points.yaml have input channel 3, which as I understand it represents the normals at the leaf octants, i.e. nx,ny,nz. In the autoencoder however,  ae_resnet.yaml specifies input channel as 4, but using my ae_ocnn.yaml I get error if I don't set input channel to 3. Specifically I get the error ```F octree_property_op.cc:101] Check failed: channel_ == channel (4 vs. 3)The specified channel_ is wrong.```. I thought I might have to generate adaptive octrees, so I repeated **step 2** with an additional argument ```--adaptive 4```. However, running the ocnn autoencoder showed again the same error. I dont understant why I get the error, and I'm also wondering why the cls_points.yaml has model channel 3.

**Q2.** What does the output of check_octree means:

example of an octree generated with ```octree ... --depth 6 --split_label 1 --rot_num 6```:
```
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
```
example of an octree generated with ```octree ... --depth 6 --split_label 1 --rot_num 6 --adaptive 4```:
```
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
```

**Q2.1.** i guess adaptive_layer:4 is dummy in the first example because of is_adaptive: 0 ?!

**Q2.2.** what is the channel parameter and the locations parameter showing?

**Q3.** Lastly, trying to understand the octree_property function, I run the following code:

```
import sys

sys.path.append("../..")
from src.data_parsing import *
from libs import *


class DatasetDebug:
    classification_channels = {'split': 0, 'label': 0, 'feature': 3, 'index': 1, 'xyz': 1}
    shape_completion_channels = {'split': 1, 'label': 1, 'feature': 3, 'index': 1, 'xyz': 1}

    @staticmethod
    def check_properties():
        # filename = 'resources/ModelNetOnly4Samples3/m40_5_2_12_test_octree.tfrecords'
        # octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
        # octree, label, filename = octrees(filename, 32, shuffle_size=False, return_iterator=False, take=10)
        # filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
        # octrees = OctreeDatasetDebug(ParseExampleDebug(x_alias='data', y_alias='label'))
        # octree, label, filename = octrees(filename, 32, shuffle_size=False, return_iterator=False, take=10)

        # filename = '/media/christina/Data/ANFASS_data/O-CNN/ModelNet40/m40_test_points.tfrecords'
        # depth = 5
        filename = '/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion/completion_test_points.tfrecords'
        depth = 6
        octrees = Point2OctreeDataset(ParseExampleDebug(x_alias='data', y_alias='label'),
                                      TransformPoints(distort=False, depth=depth, offset=0.55, axis='z', scale=0.0,
                                                      jitter=0.0, angle=[180, 180, 180],
                                                      bounding_sphere=bounding_sphere),
                                      Points2Octree(depth=depth))
        octree, label = octrees(filename, batch_size=1, shuffle_size=False, return_iterator=False, take=10)
        channels_dict = DatasetDebug.classification_channels

        with tf.Session() as sess:
            # gives error if channel!=0
            for d in range(-10, 10):
                property_name = 'split'
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.float32))
                print("depth {} {} {}".format(d, property_name, result.shape))
                property_name = 'label'
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.float32))
                print("depth {} {} {}".format(d, property_name, result.shape))

            # gives error if channel!=3
            property_name = 'feature'
            for d in range(1, depth + 1):
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.float32))
                print("depth {} {} {}".format(d, property_name, result.shape))

            result = sess.run(octree_property(octree, property_name=property_name, depth=-6,
                                              channel=channels_dict[property_name], dtype=tf.float32))
            print("depth {} {} {}".format(d, property_name, result.shape))

            # gives error if channel!=1
            property_name = 'index'
            for d in range(-1, 8):
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.int32))
                print("depth {} {} {}".format(d, property_name, result.shape))

            # gives error if channel!=1
            property_name = 'xyz'
            for d in range(-1, 8):
                result = sess.run(octree_property(octree, property_name=property_name, depth=d,
                                                  channel=channels_dict[property_name], dtype=tf.uint32))
                print("depth {} {} {}".format(d, property_name, result.shape))


DatasetDebug.check_properties()
```

which gives:
```
depth -10 split (0, 1596993073)
depth -10 label (0, 1596993073)
depth -9 split (0, 0)
depth -9 label (0, 0)
depth -8 split (0, 1)
depth -8 label (0, 1)
depth -7 split (0, 6)
depth -7 label (0, 6)
depth -6 split (0, 2)
depth -6 label (0, 2)
depth -5 split (0, 4)
depth -5 label (0, 4)
depth -4 split (0, 0)
depth -4 label (0, 0)
depth -3 split (0, 1073741824)
depth -3 label (0, 1073741824)
depth -2 split (0, 1036831949)
depth -2 label (0, 1036831949)
depth -1 split (0, 0)
depth -1 label (0, 0)
depth 0 split (0, 1)
depth 0 label (0, 1)
depth 1 split (0, 8)
depth 1 label (0, 8)
depth 2 split (0, 64)
depth 2 label (0, 64)
depth 3 split (0, 160)
depth 3 label (0, 160)
depth 4 split (0, 568)
depth 4 label (0, 688)
depth 5 split (0, 1696)
depth 5 label (0, 1952)
depth 6 split (0, 7472)
depth 6 label (0, 7016)
depth 7 split (0, 0)
depth 7 label (0, 0)
depth 8 split (0, 0)
depth 8 label (0, 0)
depth 9 split (0, 0)
depth 9 label (0, 0)
depth 1 feature (3, 8)
depth 2 feature (3, 64)
depth 3 feature (3, 176)
depth 4 feature (3, 464)
depth 5 feature (3, 1768)
depth 6 feature (3, 5080)
depth 6 feature (3, 2)
depth -1 index (1, 0)
depth 0 index (1, 1)
depth 1 index (1, 8)
depth 2 index (1, 64)
depth 3 index (1, 192)
depth 4 index (1, 552)
depth 5 index (1, 1704)
depth 6 index (1, 6784)
depth 7 index (1, 0)
depth -1 xyz (1, 0)
depth 0 xyz (1, 1)
depth 1 xyz (1, 8)
depth 2 xyz (1, 64)
depth 3 xyz (1, 192)
depth 4 xyz (1, 576)
depth 5 xyz (1, 1968)
depth 6 xyz (1, 7016)
depth 7 xyz (1, 0)
```

**Q3.1.** what is split ?

**Q3.2.** i guess in all properties i should ignore anything that is called with a non existing depth.

for classification we don't have label for each voxel that's why channel = 0 and result arrays have zero rows. (in SHAPE COMPLETION / partnet segmentation DATA CHANNEL SHOULD BE 1!?)

Q8. is the "input signal" the "feature" of octree_property and accordingly you can set it to be the result of a convolution? and **when we apply a convolution and call octree_property with this i will get a new result in each training step?**


Q4. I'm confused with the use of 'points' vs 'octree'. In the classification code, using 'octree' calls the DatasetFactory that reads from octree tfrecords, and using 'points' calls the DatasetFactory that reads from point tfrecords , transforms them and merges them to octrees. So is 'merge octree' a function that can run either on ".octree" or on ".points" and can generate either merged ".octrees" or merged ".points" accordingly? Both formats are in bytes and i can't see their difference and i see that is possible to call 
 the octree_property function on either data loaded from ".points" file or ".octree" files...

the octree2mesh ... if i have a predicted octree from OCNN i will see equal size patches and if i predicted octree from AOCNN i will see different-size patches?!

the autoencoder results are both input and output saved as ".octree" even if we use resnet or ocnn ... why/how
why input of ocnn vs resnet are slightly different?
