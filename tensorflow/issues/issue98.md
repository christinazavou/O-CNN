- **Q1**: is there a way to see the tensorflow implementation of occn_conv ?

- **Q2**: in the papers it says "since each model is rotated to 12 poses, in the testing phase the activations of the output layer for each pose can be pooled together to increase the accuracy of predictions." and this is done for classification, retrieval, segmentation ... where can i see this in the code (i.e. in test phase and not in train phase)

- **Q3**: in segmentation (in OCNN paper) they optionally do a refinement after prediction of each point .. is this done by default in Caffe/tensorflow ?

- **Q4**: if i use aocnn instead (in segmentation) will i be able to run bigger batch ? TODO

- **Q5**: how are predictions per point happening if we only have shape features?


- SQ1: isxiei to number of channels for Ul is set to 2^max(1,9−l) sta diktia tous? iparxei idikos logos pou einai etsi ? i apla epeidi genika se CNN me images aspoume oso mikrainei to resolution (widthxheight) toso auksanontai ta channels? TODO: check cls, ae, seg channels per level

    classification:
        RESNET:
            conv1:
            octree_resblock: depth 5, channel input is 64 (2^6) and channel output is 16 (2^4 i.e. 2^9-5) SO YES
            octree_resblock: depth 5, channel input is 64 and output also 64
        OCNN:
            maybe it is only here .. because it says Ul where Ul is defined as simple convolution - relu - pooling == >TODO: check this
            indeed in here they use channels = [512, 256, 128, 64, 32, 16, 8, 4, 2]
            thus in depth 5 the output channels are 2^4
            in depth 4 the output channels are 2^5
            
            question: after max pool it means that depth 5 gives to depth 4 an input of 2^3 ? TODO: check this
        
        HRNET:
    
    autoencoder:
    
    segmentation:
        in depth 7 with input 128 we will output 16.0 channels (2^4 = 2^9-5)
        in depth 6 with input 128 we will output 32.0 channels (2^5 = 2^9-4)

- SQ2: i dont understant "This is because the indicator function that represents the original shape is defined in a volume, while our octree is built from the point cloud. After replacing the normal signal as the occupying bits, it is equivalent to discarding the inside portion of the indicator function, which causes information loss compared with the full voxel representation and makes it hard to distinguish the inside and outside of the object"

- SQ3: in aocnn paper: "To make the normal direction consistent to the underlying shape normal, we check whether the angle between n and the average normals of SO is less than 90 degrees: if not, n and d are multiplied by −1" is it because the plane can have a normal in two opposite directions so we get the normals of the points to find which direction it is?

- SQ4: in aocnn paper: not sure what is "the edge length of the finest grid of the octree". not sure what is the difference of fig3middle and 3right.

- SQ5: in aocnn paper: "The loss function of the Adaptive O-CNN decoder includes the structure loss and the patch loss." .. is this in TF code? (i guess yes in shape completion network)

- SQ6: in midnet paper: they say they do "two shape segmentation". do they mean "part segmentation"? or what? which **two**? maybe they mean the part segmentation using point-loss and the part segmentation using shape-loss? 


- N1: in fig6 they show what each layer's activations focus on...is this in the code? i guess no but we can add it

- N2: in segmentation (in OCNN paper) they reuse the weights trained by the retrieval task. (the **convolution part is initialized with the weight trained on ShapeNet and fixed during optimization**, while the **weight of the deconvolution part is randomly initialized and then evolves** according to the optimization process).

- N3:in aocnn paper it says "models the 3D shape within each octant with a planar patch ... it takes the planar patch and displacement as input" ... i think this is what tensorflow implementation takes in all tasks ...  the default ocnn takes just nx,ny,nz without d, but in all TF implementations they used nx,ny,nz,d

- N4: "We pre-process the point cloud to assign a normal vector for each point via principal component analysis if the accurate normal information is not available in the dataset, which will be used for constructing input features." i guess in ```new_points()``` there is pca if normals are not specified .. or there was somewhere else this step and in new_points() is mandatory to give normals..



- "we first exploit the shape instance discrimination loss to classify the point of each shape into a class and then apply the point instance discrimination to classify the points on each shape separately" how is this in the code ?

- in midnet paper: "we convert each normal component to its absolute value" this probably happens in octree creation...so if i give points without absolute normal and then i call octree_property('feature') i guess it will output absolute normal.. right? TODO: write a unit test

- in midnet paper: "For point-wise features, we interpolate the high-resolution features defined at the second finest-level octants according to the point position via tri-linear interpolation." is this happening also in ocnn paper's segmentation ? what does it really mean ? that features at level maxdepth-1 = interpolation of features at maxdepth level from neighbour octants?

- in midnet paper: "Note that the high-resolution features at each octant are also the concatenation of features obtained from the high-resolution network and the upsampled feature from the low-resolution subnetwork." edo ti ennoei ? ennoei ta features meta ta convolution kai activation ? 

- in midnet .. ti ginetai me to point loss an ena shape ginei classified se lathos shape class?

- feature visualization of shape-level and point-wise features with tsne...not in code?

- afou kano apla segmentation with random init ... ti loss function xrisimopoiei ? to mid loss i kapoio allo ?

- sto midnet paper: MID-FC(Fix) ti xrisimopoioun gia input sto FC? (note: look at fig 2 .. where final feature vectors for points have 926 channels and final feature vectors for shapes have 896 channels )

- "After that, the output **shape or point** features are fed into the following back end layers for computing the shape analysis results." ... ara mipos using the flag LOSS.point_wise as false tha xrisimopoiei ta shape features gia to segmentation eno using True tha xrisimopoiei ta point features?

- is there a configuration to run segmentation with mid loss or is it only with shape loss or point loss ?
