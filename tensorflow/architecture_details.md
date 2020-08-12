the O-CNN takes the average normal vectors of a 3D model sampled in the finest leaf octants as input and computes features for the finest level octants. After pooling, the features are down-sampled to the parent octants in the next coarser level and are fed into the next O-CNN layer. This process is repeated until all O-CNN layers are evaluated

we pack the features and data of sparse octants at each depth as continuous arrays. A label buffer is introduced to find the correspondence between the features at different levels for efficient convolution and pooling operations.

##### input to the OCNN:
 - an oriented 3D model (e.g. an oriented triangle mesh or a point cloud with oriented normals)

##### To construct an octree for an input 3D model:
- we first uniformly scale the 3D shape into an axis-aligned unit 3D bounding cube and then recursively subdivide the bounding cube of the 3D shape in breadth-first order. In each step, we traverse all non-empty octants occupied by the 3D shape boundary at the current depth l and subdivide each of them to eight child octants at the next depth l + 1. We repeat this process until the pre-defined octree depth d is reached.

##### To do the first conv:
- our method extracts the input signal of the CNN from the 3D shape stored in the finest leaf nodes and records the resulting CNN features at each octant.

##### Properties defined in the octree structure:

1.	**Shuffle key**:
	The shuffle key of an octant O at depth l encodes its position in 3D space with an unique 3 l-bit string key.
2. 	**Label**:
	we assign a label p for a non-empty octant at the l-th depth, which indicates that it is the p-th non-empty octant in the sorted octant list of the l-th depth.
3. 	**Input signal**:
	We use the averaged normal vectors computed at the finest leaf octants as the input signal of the CNN.
4.	**CNN features**:
	For each 3D convolution kernel defined at the l-th depth, we record the convolution results on all the octants at the l-th depth in a feature map vector Tl.
5.	**Mini-batch of 3D models:**
	For 3D objects in a mini-batch used in the CNN training, their octrees are not the same. To support efficient CNN training on the GPU, we merge these octrees into one superoctree.

#####CNN operations on the octree:

1.	**3D convolution**:
	A convolution with a stride of 2^r can be applied to the first octant belonging to each sub-tree of height r, then the feature map will be downsampled by a factor of 2^r.
	Because of the special hierarchical structure of an octree, the stride of convolution is constrained to be an integer power of 2.
2.	**Pooling**:
	Applying the max-pooling operator on an octree reduces to picking out the max elements from every 8 contiguous element.
	Then the resolution of the feature map is down-sampled by a factor of 2.
3.	**Unpooling**:
	After applying the max-pooling operation, the locations of the maxima within each pooling region can be recorded in a set of switch variables stored in a continuous array. The corresponding max-unpooling operation makes use of these switches to place the signal of the current feature map into appropriate locations of the up-sampled feature map
4.	**Deconvolution**:
	can be implemented by just reversing the forward and backward passes of convolution

##### Network structure:

###### OCNN:
	
- We repeatedly apply convolution and pooling on the octree data structure from bottom to top. We use the ReLU function to activate the output and use batch normalization to reduce the  internal-covariateshift
- “convolution + BN + ReLU + pooling”  = a basic unit = Ul
- The number of channels of the feature map for Ul is set to 2^max(1,9−l) and the convolution kernel size is 3.
- We enforce all the 2nd-depth octants to exist and use zero vector padding on the empty octants at the 2nd depth.
- input → Ud → Ud−1 → · · · → U2 = OCNN(d)

###### OCNN for object classification:
- We add two fully connected (FC) layers, a softmax layer, and two Dropout layers after OCNN(d)
- O-CNN(d) → Dropout → FC(128) → Dropout → FC(Nc ) → softmax → output

###### OCNN for shape retrieval:
- Same as for object classification. Output vector used for similarity.

###### OCNN for part segmentation:
- The convolution network is set as our O-CNN(d).
- The deconvolution network is the mirror of O-CNN(d) where the convolution and pooling operators are replaced by deconvolution and unpooling operators.
- “unpooling + deconvolution + BN + ReLU” = a basic unit = DUl
- O-CNN(d) → DU2 → DU3 → · · · → DUd

##### Experiments:
- a desktop machine with an Intel Core I7-6900K CPU (3.2 GHz) and a GeForce 1080 GPU (8GB memory)
- stochastic gradient descent (SGD) with a momentum of 0.9, a weight decay of 0.0005, and a batch size of 32. The dropout ratio is 0.5. The initial learning rate is 0.1, and decreased by a factor of 10 after every 10 epochs. The optimization stops after about 40 epochs.

##### Dataset:
- ModelNet40 and ShapeNetCore55
- to build the octree data structure with correct normal information, we first use the ray shooting algorithm to sample dense points with oriented normals from the shapes. Specifically, we place 14 virtual cameras on the face centers of the truncated bounding cube of the object, uniformly shoot 16k parallel rays towards the object from each direction, calculate the intersections of the rays and the surface, and orient the normals of the surface points towards the camera. The points on the invisible part of the shapes are discarded. We then build an octree structure on the point cloud and compute the average normal vectors of the points inside the leaf octants. (_Virtual_Scanner_)

###### object classification:
- ModelNet40: 12,311 CAD models from 40 categories with multi-class labels
- 9,843 models are used for training, and 2,468 models for testing
- The upright orientation of the models in the dataset is known. We augment the dataset by rotating each model along the upright direction uniformly to generate 12 poses for each model.
- in the test phase we use orientation pooling

###### shape retrieval:
- ShapeNet Core55: 51190 3D models with 55 categories and 204 subcategories
- The models are normalized to a unit length cube and have a consistent upright orientation. 70% of the dataset is used for training, 10% for validation, and 20% for testing. We use the same method as object classification to perform data augmentation.
- In the training stage, the cross-entropy loss function is minimized with only the category information. The subcategory information in the dataset is discarded for simplicity
- orientation pooling is used to generate one feature vector for each shape
- The retrieval set of a query shape is constructed by collecting all shapes that have the same label, and then sorting them according to the feature vector distance between the query shape and the retrieved shape

###### object part segmentation:
- The goal is to assign part category information to each point or triangle face
- ShapeNet: 16 categories of shapes, with 2 to 6 parts per category. In total there are 16,881 models with part annotations.
- sparse point clouds, with only about 3k points for each model, and the point normals are missing
- We align the point cloud with the corresponding 3D mesh, and project the point back to the triangle faces. Then we assign the normal of the triangle face to the point, and condense the point cloud by uniformly re-sampling the triangle faces. Based on this pre-processed point cloud, the octree structure is built
- 12 copies rotated around the upright axis. 
- limited data ==> thus we reuse the weights trained by the retrieval task
- Specifically, in the training stage, the convolution part is initialized with the weight trained on ShapeNet and fixed during optimization, while the weight of the deconvolution part is  andomly initialized and then evolves according to the optimization process


note in TF implementation:
** probably they use two graphs, one with train and one with test because:
1. in train they want to support multi gpu thus reuse=True is useful
2. using TFRecordDataset its easy to make one dataset reading training data and one reading input data and then just pass them as inputs .. instead of feeding training data and then switching dataset to read test...
