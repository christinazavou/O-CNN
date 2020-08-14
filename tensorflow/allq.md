Hello again,

I'm trying to run the Tensorflow implementation for the autoencoder and I have a few questions.

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

3. similarly to filelist_test.txt and filelist_train.txt I generated files that include paths of the corresponding octrees (for each .points file path i included the 6 .octree file paths) and then I made octree tfrecords. specifically the corresponding files:
```
completion_test_octrees.tfrecords
completion_test_scans_octrees.tfrecords
completion_train_octrees.tfrecords
```

4. i run the autoencoder with 
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

What I noticed is that the resnet with points is taking more time to do the same amount of batch iterations (1 hour equals to 2800 iterations in resnet and 12000 iterations in ocnn) and the accuracy and loss of resnet are much smoother than the ones in ocnn. e.g.
![image](https://user-images.githubusercontent.com/15656466/90230817-cbf05100-de22-11ea-9852-32a255e66f74.png) vs. ![image](https://user-images.githubusercontent.com/15656466/90230846-d6124f80-de22-11ea-9d79-c563295561ca.png)
Also, the decoded octrees from resnet are much better than the ones from ocnn: e.g.
![image](https://user-images.githubusercontent.com/15656466/90230991-1e317200-de23-11ea-9d22-2f5bd2318a4a.png) vs. ![image](https://user-images.githubusercontent.com/15656466/90231021-2a1d3400-de23-11ea-8cd3-af85358dd625.png)

Except from these, I have some general questions..

However, I'm not sure I understand the difference of the code using 'points' and using 'octree'. In the classification code, using 'octree' calls the DatasetFactory that reads from octree tfrecords, and using 'points' calls the DatasetFactory that reads from point tfrecords , transforms them and merges them to octrees. So is 'merge octree' a function that can be run either on ".octree" or on ".points" and it generates either merged ".octrees" or merged ".points"? Because both formats are in bytes and i can't see their difference; Also, can I call "octree_property" function both on data loaded from ".points" file or ".octree" file but with different arguments? Which arguments can be used for each one?

Another question I have is regarding the input signal. In classification both cls_octree.yaml and cls_points.yaml have input channel 3, which as I understand it represents the normals at the leaf octants, i.e. nx,ny,nz. In the autoencoder however,  ae_resnet.yaml specifies input channel as 4, but using my ae_ocnn.yaml I get error if I dont set input channel to 3. This rises two questions for me:
1. is the code using cls_octree.yaml / cls_points.yaml running OCNN or AOCNN? (because the input in AOCNN is supposed to be (n,d*) i.e. 4-dimensional)
2. should I use --adaptive true when generating the completion octrees in order to use input channel 4?
3. where the octrees generated for classification using adapt false and that's why input channel was 3?
4. with the .points data and the resnet can we use any input channel ?


AOCNN paper: from fig4 i see the encoder having skip_connections ... are these the ones referred in the code (where there is flag to use skip_connections)  or not !?

also wondering what does the output of check_octree means:

e.g.
This is a valid octree!
magic_str:_OCTREE_1.0_
batch_size: 1
depth: 5
full_layer: 2
adaptive_layer: 4
threshold_distance: 2
threshold_normal: 0.1
is_adaptive: 0
has_displace: 0
nnum: 1 8 64 144 432 1816 0 0 0 0 0 0 0 0 0 0 
nnum_cum: 0 1 9 73 217 649 2465 2465 0 0 0 0 0 0 0 0 
nnum_nempty: 1 8 18 54 227 836 0 0 0 0 0 0 0 0 0 0 
total_nnum: 2465
total_nnum_capacity: 2465
channel: 1 1 0 3 0 0 
locations: -1 -1 0 5 0 0 
bbox_max: 563.252 1111.51 730.384 
bbox_min: -571.313 -23.054 -404.181 
key2xyz: 0
sizeof_octree: 42228

e.g. why is channel always of size 6 ??

can i get "shuffle key" from octree_property?
is "label" in "octree_property" giving the "which indicates that it is the p-th non-empty octant in the sorted octant list of the l-th depth." ?
can i get "input signal" from octree_property? is it the "xyz" property or the "feature" property ?
can i get "cnn features" from octree_property? if i apply a convolution and call again octree_property with this i will get a new result in each training step?

displace = translate the location of the object ? that's why in input/output of decoder we have different locations ?

the octree2mesh ... if i have a predicted octree from OCNN i will see equal size patches and if i predicted octree from AOCNN i will see different-size patches?!

what is the 4-dim input signal ? why autoencoder with points wants channel 4 and with octrees wants channel 3 ?

the autoencoder results are both input and output saved as ".octree" even if we use resnet or ocnn ... why/how

