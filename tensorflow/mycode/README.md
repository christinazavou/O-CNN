
## OCNN classification

input: DatasetFactory returns octree, label, filename. We use octree as input and we use label to calculate the loss.

## OCNN autoencoder

input: DatasetFactory returns octree, label, filename. We use octree as input

(Both points datafactory and octrees datafactory generate octrees.)

### OCNN segmentation

##### run_seg_partnet.py

touto xrisimopoiei to HRNet (i to UNet) gia na kamei segmentation diladi gia kathe shape category prepei na trekseis ena run_seg_partnet me training kai test data mono autis tis katigorias. Touto ousiastika einai to encoder (pou sindiazei shape features kai point features) me ena linear classification (FC layer) sto telos gia na kamei predict ta parts apo ekeini tin katigoria.

##### run_seg_partnet_cmd.py

touto xrisimopoiei to gia na treksei kamposa peiramata, diladi me diafora amount of training data (parameter 'ratio') kai gia kathe shape category. dame exeis to option eite na kalei to run_seg_partnet.py (opou to HRNet - o encoder) tha ginei randomized initialized eite na kalei to run_seg_partnet_finetune.py (check below) 

##### run-seg_partnet_finetune.py

touto einai to idio me to run_seg_partnet.py aplos ola ta weights tou encoder ginontai initialized vasei tou pretrained diktiou to opoio egine pretrained me MID loss.

gia na to trekseis touto katevazeis to pretrained model (Einai HRNet - logika gia UNet prepei na kamoume emeis to pretraining. Episis katevenoun ta data pou to ekanan trained.) to opoio egine trained me to run_mid.py

##### note:

to MIDNet einai ousiastika ena diktio pou apoteleite apo ena encoder kai sto telos ena classifier layer gia kathe shape category kai ena classifier layer gia shape category. Gia na ginei trained thelei alla data - ta opoia logika einai shapes apo mixed categories mazi me ta parts tous - kai kata to training xrisimopoiei ena loss pou apoteleite apo to loss tou category kai to loss ton points kai kanei update to encoder me diaforetiko rithmo apo ta classifier layers


### To be checked

- to label_gt > mask where mask = -1 ara label_gt opos to pairno apo octree_property einai -1 or 1 ?!
- ta points sto train einai "osa octrees exo sto batch enono ta points tous kai meta perno ta misa points" eno sto test einai "ena octree pou exei 1000 points kai ta perno ola"
- ola ta intersections auksithikan extos to intersection 1 ... poia klassi einai ?
- ta num_class einai panta to amount of level3 categories plus 1(for undefined)
- exoume kanena diktio pou kanei kai classification kai segmentation mazi? episis na do pos kanei to pretrain (i.e. run mid.py)
 - kai to preprocessing tou shapenet einai pali tfrecords per shape category .. opotan kaneis ena diktio gia kathe category .. mono gia to midnet exei na katevaseis ena tfrecords file pou mallon exei diafora shapes mazi !..


### data/seg_partnet:

- convert ply: 

    takes the folder of annotated sample which contains the labels in one file, the points in another, the ply with colors according to labels


note:
- https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
(sto IoU ola ta points einai idiou importance)
https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
