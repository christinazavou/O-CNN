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

train-seg-hrnet-buildings-no-colour:
	export PATH=/home/graphicslab/miniconda3/bin/:$$PATH \
	&& source activate OCNN \
	&& cd .. \
	&& python run_seg_partnet.py \
		--config configs/segmentation/seg_hrnet_partnet_pts.yaml \
		SOLVER.run train \
		SOLVER.gpu 0, \
		SOLVER.logdir /media/graphicslab/Elements/ANNFASS_data/seg_out/debugD7NoC \
		SOLVER.ckpt '' \
		SOLVER.max_iter 1000 \
		SOLVER.step_size 500,250 \
		SOLVER.test_every_iter 200 \
		DATA.train.location /media/graphicslab/Elements/ANNFASS_data/seg_data/no_colour/train_points100000.tfrecords \
		DATA.train.depth 7 \
		DATA.train.batch_size 4 \
		DATA.train.take 1600 \
		DATA.test.location /media/graphicslab/Elements/ANNFASS_data/seg_data/no_colour/test_points100000.tfrecords \
		DATA.test.depth 7 \
		DATA.test.batch_size 4 \
		MODEL.nout 34 \
		LOSS.num_class 34 \
		MODEL.depth 7 \
		MODEL.depth_out 7

params-seg-hrnet-buildings-no-colour:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd .. \
		&& python run_seg_partnet.py \
			--config configs/segmentation/seg_hrnet_partnet_pts.yaml \
			SOLVER.run param_stats \
			SOLVER.gpu 0, \
			SOLVER.logdir /media/christina/Elements/ANNFASS_DATA/output/segmentation/Building/hrnet_randinit/Batch4NoColourDepth6 \
			SOLVER.ckpt /media/christina/Elements/ANNFASS_DATA/output/segmentation/Building/hrnet_randinit/Batch4NoColourDepth6/model/iter_001000.ckpt \
			DATA.train.location /media/graphicslab/Elements/ANNFASS_data/seg_data/no_colour/train_points100000.tfrecords \
			DATA.train.batch_size 4 \
			DATA.train.take 1600 \
			DATA.test.location /media/graphicslab/Elements/ANNFASS_data/seg_data/no_colour/test_points100000.tfrecords \
			DATA.test.batch_size 4 \
			MODEL.nout 34 \
			LOSS.num_class 34

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

train-cls-modelnet40-resnet:
	export PATH=/home/graphicslab/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd .. \
		&& python run_cls.py --config configs/cls/cls_points.yaml \
		SOLVER.run train \
		SOLVER.gpu 0, \
		SOLVER.logdir /media/graphicslab/Elements/ANNFASS_data/cls_out/debugResnet \
		SOLVER.ckpt '' \
		DATA.train.location /media/graphicslab/Elements/ANNFASS_data/cls_data/m40_train_points.tfrecords \
		DATA.test.location /media/graphicslab/Elements/ANNFASS_data/cls_data/m40_test_points.tfrecords

train-cls-modelnet40-ocnn:
	export PATH=/home/graphicslab/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd .. \
		&& python run_cls.py --config configs/cls/cls_points.yaml \
		SOLVER.run train \
		SOLVER.gpu 0, \
		SOLVER.logdir /media/graphicslab/Elements/ANNFASS_data/cls_out/debugOcnn \
		SOLVER.ckpt '' \
		MODEL.name ocnn \
		DATA.train.location /media/graphicslab/Elements/ANNFASS_data/cls_data/m40_train_points.tfrecords \
		DATA.test.location /media/graphicslab/Elements/ANNFASS_data/cls_data/m40_test_points.tfrecords

test-cls-modelnet40-ocnn:
	export PATH=/home/graphicslab/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd .. \
		&& python run_cls.py --config configs/cls/cls_points.yaml \
		SOLVER.run test \
		SOLVER.gpu 0, \
		SOLVER.logdir /media/graphicslab/Elements/ANNFASS_data/cls_out/debugOcnn \
		SOLVER.ckpt /media/graphicslab/Elements/ANNFASS_data/cls_out/debugOcnn/model/iter_001000.ckpt \
		MODEL.name ocnn \
		DATA.train.location /media/graphicslab/Elements/ANNFASS_data/cls_data/m40_train_points.tfrecords \
		DATA.test.location /media/graphicslab/Elements/ANNFASS_data/cls_data/m40_test_points.tfrecords

train-seg-partnet-hrnet-bottle:
	--config configs/segmentation/seg_hrnet_partnet_pts.yaml SOLVER.run train SOLVER.gpu 0, SOLVER.logdir /media/christina/Data/ANFASS_data/seg_out/Bottle/debugDef SOLVER.test_iter 84 SOLVER.ckpt '' DATA.train.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Bottle_train_level3.tfrecords DATA.test.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Bottle_test_level3.tfrecords MODEL.nout 9 MODEL.factor 2 LOSS.num_class 9 DATA.train.take 315 DATA.test.batch_size 1 DATA.train.batch_size 4

train-seg-hrnet-chair:
	python run_seg_partnet.py --config configs/segmentation/seg_hrnet_partnet_pts.yaml SOLVER.run train SOLVER.gpu 0, SOLVER.logdir /media/graphicslab/BigData/zavou/ANNFASS_data/seg_out/Chair/hrnet SOLVER.max_iter 20000 SOLVER.test_iter 1217 SOLVER.ckpt '' DATA.train.location /media/graphicslab/BigData/zavou/ANNFASS_data/seg_data/partnet/Chair/Chair_train_level3.tfrecords DATA.test.location /media/graphicslab/BigData/zavou/ANNFASS_data/seg_data/partnet/Chair/Chair_test_level3.tfrecords MODEL.nout 39 MODEL.factor 2 LOSS.num_class 39 DATA.train.take 4489 DATA.test.batch_size 1 DATA.train.batch_size 4
