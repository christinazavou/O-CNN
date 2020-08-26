SHELL := /bin/bash # otherwise we can't use "source"

utils = "/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/util"
scripts = "/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script"

params-cls-octree:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${utils} \
		&& python parse_ckpt.py --ckpt ${scripts}/logs/m40/ocnn_octree5_b32/model/iter_068000.ckpt

params-cls-octree-mine:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${utils} \
		&& python parse_ckpt.py --ckpt /media/christina/Data/ANFASS_data/O-CNN/output/ModelNet40/logsT3/model/iter_066000.ckpt

params-cls-points-nodedis:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${utils} \
		&& python parse_ckpt.py --ckpt /media/christina/Data/ANFASS_data/O-CNN/output/ModelNet40/m40/aresnet_b32/model/iter_142000.ckpt

params-ae-points:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${utils} \
		&& python parse_ckpt.py --ckpt /media/christina/Data/ANFASS_data/O-CNN/output/ocnn_completion/ae/resnet_b16/model/iter_006000.ckpt

params-ae-octree:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${utils} \
		&& python parse_ckpt.py --ckpt /media/christina/Data/ANFASS_data/O-CNN/output/ocnn_completion/ae/ocnn_b16/model/iter_020000.ckpt

count-data:
	/bin/bash count_data.sh > partnetdata.txt
