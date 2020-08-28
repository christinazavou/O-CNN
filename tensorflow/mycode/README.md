
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

- mallon to run_seg_partnet_cmd an den einai finetune tote apla xtizei 17 kainouria networks .. ta opoia kanoun segmentation ta patches tou object .. diladi emeis gia kathe arxitektoniko tipo tha theloume ena tetoio diktio .. kai an valo finetune kanei load to pretrained encoder kai randomized initialization sto segmentation header i.e. to final layer i.e. to sigkekrimeno classifier ... (opotan to loss den einai to MID loss sto run_seg_partnet alla to cross entropy ton part predictions) .. so aporia: exoume kanena diktio pou kanei kai classification kai segmentation mazi? episis na do pos kanei to pretrain (i.e. run mid.py)
 
 - kai to preprocessing tou shapenet einai pali tfrecords per shape category .. opotan kaneis ena diktio gia kathe category .. mono gia to midnet exei na katevaseis ena tfrecords file pou mallon exei diafora shapes mazi !..
