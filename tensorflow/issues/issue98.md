- is there a way to see the tensorflow implementation of occn_conv ?

- isxiei to number of channels for Ul is set to 2^max(1,9−l) sta diktia tous? iparxei idikos logos pou einai etsi ? i apla epeidi genika se CNN me images aspoume oso mikrainei to resolution (widthxheight) toso auksanontai ta channels?

- in object classification ... "Since each model is rotated to 12 poses, in the testing phase the activations of the output layer for each pose can be pooled together to increase the accuracy of predictions." ... can i see in the code difference in train mode and test mode regarding this ? (it is also done in shape retrieval...is shape retrieval in the available code? it is also done in part segmentation .. can i see it there?)

- i dont understant this "This is because the indicator function that represents the original shape is defined in a volume, while our octree is built from the point cloud. After replacing the normal signal as the occupying bits, it is equivalent to discarding the inside portion of the indicator function, which causes information loss compared with the full voxel representation and makes it hard to distinguish the inside and outside of the object"

- in fig6 they show what each layer's activations focus on...is this in the code? i guess no but we can add it

- in segmentation (in OCNN paper) they reuse the weights trained by the retrieval task.
(the **convolution part is initialized with the weight trained on ShapeNet and fixed during optimization**, while the **weight of the deconvolution part is  randomly initialized and then evolves** according to the optimization process)

- in segmentation (in OCNN paper) they optionally do a refinement after prediction of each point .. is this done by default in Caffe/tensorflow ?

