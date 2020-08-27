
## OCNN classification

input: DatasetFactory returns octree, label, filename. We use octree as input and we use label to calculate the loss.

## OCNN autoencoder

input: DatasetFactory returns octree, label, filename. We use octree as input

(Both points datafactory and octrees datafactory generate octrees.)

### OCNN segmentation

- to label_gt > mask where mask = -1 ara label_gt opos to pairno apo octree_property einai -1 or 1 ?!
- ta points sto train einai "osa octrees exo sto batch enono ta points tous kai meta perno ta misa points" eno sto test einai "ena octree pou exei 1000 points kai ta perno ola"
- ola ta intersections auksithikan extos to intersection 1 ... poia klassi einai ?
- ta num_class einai panta to amount of level3 categories plus 1(for undefined)
