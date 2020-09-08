
Our method represents a 3D shape adaptively with octants at different levels and models the 3D shape within each octant with a planar patch

In existing work, the non-empty leaf octants at the finest level can be regarded as uniform samples of the shape in the x, y, and z directions. We observe that it is actually not necessary to store the shape information in this uniform way since the local shape inside some octants can be represented by simple patches, like planar patches.

Therefore, by storing the patch information and terminating the octant split early if the patch associated with the octant well approximates the local shape, the generated octree can have a more compact and adaptive representation. 

The stored patch has a higher order approximation accuracy than using the center or one of the corners of the octant as the sample of the surface.

**_AOCNN adaptively splits the octant according to the approximation error of the estimated simple patch to the local shape contained by the octant.**_ 

todo: move this note to annfass documentation
note:
These point-based CNNs (PointNet, PointNet++) are suited to applications whose input can be well approximated by a set of points or naturally has a point representation, like LiDAR scans. For mesh inputs where the neighbor region is well-defined, graph-based CNNs and manifold-based CNNs find their unique advantages for shape analysis.

#### Patch guided adaptive octree

1. For a given surface S, we start with its bounding box and perform 1-to-8 subdivision
2. For octant O, denote _SO as the local surface of S restricted by the cubical region of O_. 
    - If SO != ∅, we approximate a simple surface patch (a planar patch) to SO.
    - The best plane P with the least approximation error to SO is the minimizer of the following objective: integral of ||np+d||^2 (n ∈ R^3 is the unit normal vector of the plane and the plane equation is P: n · x + d = 0, x ∈ R^3)
    - To make the normal direction consistent to the underlying shape normal, we check whether the angle between n and the average normals of SO is less than 90 degrees: if not, n and d are multiplied by −1
3. We denote _PO as the planar patch of P restricted by the cubical region of O_. The shape approximation quality of the local patch, δO, is defined by the Hausdorff distance between PO and SO
4. The revised partitioning rule of the octree is: For any octant O which is not at the max depth level, subdivide it if SO != ∅ and δO is larger than the predefined threshold ˆδ.

note:
ˆδ is set as (√3/2)h, where h is the edge length of the finest grid of the octree

#### Adaptive OCNN

There are two major differences between Adaptive O-CNN
and O-CNN: 

1. the input signal appears at all the octants, not only at the finest octants; 
2. the computation starts from leaf octants at different levels simultaneously and the computed features are assembled across different levels.

- Input signal:
    
    For an octant O at the l-level whose local plane is n·x+d=0, x ∈ R^3, we set a four-channel input signal in it: (n,d⋆). 
    
    Here c is the center point of O and d⋆ = d−n·c. Note that n·(x−c) + d⋆ = 0 is the same plane equation.

    Here we use d⋆ instead of d because d⋆ is bounded by the grid size of l-level and it is a relative value while d has a large range since d measures the distance from the origin to the plane.

    For an empty octant, its input signal is set to (0, 0, 0, 0). 

##### Adaptive OCNN 3D encoder
On each level of the octree, we apply a series of convolution operators and ReLUs to the features on all the octants at this level and the convolution kernel is shared by these octants. 

**Then the processed features at the l-th level is downsampled to the (l − 1)-th level via pooling and are fused with the features at the (l−1)-th level by the element-wise max operation.**
(i.e. i understand that features are downsampled and fused before convolutions happen.)

##### 3D decoder of AOCNN

The output of the prediction module:
 - includes the patch approximation status and the plane patch parameters (n,d⋆).

The loss function of the Adaptive O-CNN decoder:
 - includes the structure loss and the patch loss.
 - The structure loss Lstruct measures the difference between the predicted octree structure and its ground-truth. Since the determination of octant status is a 3-class classification, we use the cross entropy loss to define the structure loss. 
 - Lstruct = sum {(wl/nl)Hl} where nl is number of octants at the lth level of the predicted octree, wl is the weight defined on each level. (we use wl=1)
 - Lpatch measures the squared distance error between the plane parameters and the ground truth at all the leaf octants in each level
 - Note that for the wrongly generated octants that do not exist in the groundtruth, there is no patch loss for them, and they are penalized by the structure loss only


#### Experiments

##### Shape classification

Data:
ModelNet40

Results:
- **We conclude that Adaptive O-CNN with a deeper octree tends to overfit the training data of ModelNet40**. With more training data, for instance, by rotating each training object 24 times around their upright axis uniformly, the classification accuracy can increase by 0.2% under the resolution of 1283.

##### 3D Autoencoding

Data:
ShapeNet Core v2

max depth used: 7

ablation study:
1. A vanilla octree based autoencoder. The decoder is similar to the Adaptive O-CNN decoder but with two differences: (1) the prediction module only predicts whether a given octant has an intersection with the ground-truth surface. If intersected, the
octant will be further subdivided; (2) the loss only involves the structure loss.
2. An enhanced version of O-CNN(binary). The prediction module also predicts the plane patch on each leaf node at the finest level and the patch loss is added.
3. An enhanced version of O-CNN(patch). The prediction module predicts the plane patch on each leaf node at each level and the patch loss is added.

Results:

##### Shape completion

##### Shape reconstruction from a single image

Dataset:
Shapenet core v2

