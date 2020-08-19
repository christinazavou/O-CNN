SHELL := /bin/bash # otherwise we can't use "source"

run-experiments:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
	&& source activate OCNN \
	&& python run_ae.py --config configs/ae_aocnn.yaml

#	&& python run_ae.py --config configs/ae_aresnet.yaml \
#	&& python run_cls.py --config configs/cls_points_nodedis.yaml
