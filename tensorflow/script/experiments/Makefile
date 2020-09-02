SHELL := /bin/bash # otherwise we can't use "source"

train-seg-hrnet-buildings:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
	&& source activate OCNN \
	&& cd .. \
	&& python run_seg_partnet.py --config configs/segmentation/seg_hrnet_partnet_pts.yaml SOLVER.run train SOLVER.gpu 0, SOLVER.logdir /media/christina/Data/ANFASS_data/O-CNN/output/segmentation/Building/hrnet_randinit/Batch4NoColourDepth7 SOLVER.max_iter 1000 SOLVER.step_size 150,75 SOLVER.test_every_iter 200 SOLVER.test_iter 200 SOLVER.ckpt '' DATA.train.location /media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset_points_tuesday/train_points100000.tfrecords DATA.train.take 1600 DATA.train.batch_size 4 DATA.test.location /media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset_points_tuesday/test_points100000.tfrecords MODEL.nout 34 MODEL.factor 2 LOSS.num_class 34 DATA.train.depth 7 DATA.test.depth 7 MODEL.depth 7 MODEL.depth_out 7

test-seg-hrnet-buildings:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
	&& source activate OCNN \
	&& cd .. \
	&& python run_seg_partnet.py --config configs/segmentation/seg_hrnet_partnet_pts.yaml SOLVER.run test SOLVER.gpu 0, SOLVER.logdir /media/christina/Elements/ANNFASS_DATA/output/segmentation/Building/hrnet_randinit/Batch4NoColourDepth6 SOLVER.test_iter 200 SOLVER.ckpt /media/christina/Elements/ANNFASS_DATA/output/segmentation/Building/hrnet_randinit/Batch4NoColourDepth6/model/iter_001000.ckpt DATA.test.location /media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/dataset_points_tuesday/test_points100000.tfrecords MODEL.nout 34 MODEL.factor 2 LOSS.num_class 34

run-experiments:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
	&& source activate OCNN \
	&& python run_ae.py --config configs/ae_aocnn.yaml

#	&& python run_ae.py --config configs/ae_aresnet.yaml \
#	&& python run_cls.py --config configs/cls_points_nodedis.yaml

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
