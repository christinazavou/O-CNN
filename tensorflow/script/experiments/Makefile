SHELL := /bin/bash # otherwise we can't use "source"

test-bottle-b16-hrnet:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd .. \
		&& python run_seg_partnet.py --config configs/segmentation/seg_hrnet_partnet_pts.yaml SOLVER.run test SOLVER.gpu 0, SOLVER.logdir /media/christina/Data/ANFASS_data/O-CNN/output/segmentation/experiments/Bottle/Bottle_b16_hrnet SOLVER.test_iter 84 SOLVER.ckpt /media/christina/Data/ANFASS_data/O-CNN/output/segmentation/experiments/Bottle/Bottle_b16_hrnet/model/iter_003000.ckpt DATA.test.location /media/christina/Data/ANFASS_data/O-CNN/output/segmentation/experiments/Bottle/Bottle_test_level3.tfrecords MODEL.nout 9 MODEL.factor 2 LOSS.num_class 9

test-bottle-b32-hrnet:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd .. \
		&& python run_seg_partnet.py --config configs/segmentation/seg_hrnet_partnet_pts.yaml SOLVER.run test SOLVER.gpu 0, SOLVER.logdir /media/christina/Data/ANFASS_data/O-CNN/output/segmentation/experiments/Bottle/Bottle_b32_hrnet SOLVER.test_iter 84 SOLVER.ckpt /media/christina/Data/ANFASS_data/O-CNN/output/segmentation/experiments/Bottle/Bottle_b32_hrnet/model/iter_003000.ckpt DATA.test.location /media/christina/Data/ANFASS_data/O-CNN/output/segmentation/experiments/Bottle/Bottle_test_level3.tfrecords MODEL.nout 9 MODEL.factor 2 LOSS.num_class 9
