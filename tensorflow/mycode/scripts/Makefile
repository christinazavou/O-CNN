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
	/bin/bash count_data.sh > output/partnetdata.txt

run-seg-bottle:
	 python run_seg_partnet.py --config configs/segmentation/seg_hrnet_partnet_pts.yaml SOLVER.gpu 0, SOLVER.logdir logs/seg/0811_partnet_randinit/Bottle/ratio_1.00 SOLVER.max_iter 3000 SOLVER.step_size 1500,750 SOLVER.test_every_iter 100 SOLVER.test_iter 84 SOLVER.ckpt '' DATA.train.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Bottle_train_level3.tfrecords DATA.train.batch_size 8 DATA.train.take 315 DATA.test.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Bottle_test_level3.tfrecords MODEL.nout 9 MODEL.factor 2 LOSS.num_class 9

run-seg-clock:
	 python run_seg_partnet.py --config configs/segmentation/seg_hrnet_partnet_pts.yaml SOLVER.gpu 0, SOLVER.logdir logs/seg/0811_partnet_randinit/Clock/ratio_1.00 SOLVER.max_iter 5000 SOLVER.step_size 2500,1250 SOLVER.test_every_iter 400 SOLVER.test_iter 98 SOLVER.ckpt '' DATA.train.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Clock_train_level3.tfrecords DATA.train.batch_size 8 DATA.train.take 406 DATA.test.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Clock_test_level3.tfrecords MODEL.nout 11 MODEL.factor 2 LOSS.num_class 11

run-seg-finetune-bed:
	python run_seg_partnet_finetune.py --config configs/segmentation/seg_hrnet_partnet_pts.yaml SOLVER.gpu 0, SOLVER.logdir logs/seg/partnet_finetune/Bed/ratio_1.00 SOLVER.max_iter 3000 SOLVER.step_size 1500,750 SOLVER.test_every_iter 100 SOLVER.test_iter 37 SOLVER.ckpt dataset/midnet_data/mid_d6_o6/model/iter_800000.ckpt DATA.train.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Bed_train_level3.tfrecords DATA.train.take 133 DATA.train.batch_size 8 DATA.test.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Bed_test_level3.tfrecords MODEL.nout 15 MODEL.factor 2 LOSS.num_class 15

run-seg-unet-test-bottle:
	--config configs/segmentation/seg_unet_partnet_pts.yaml SOLVER.run test SOLVER.gpu 0, SOLVER.logdir logs/seg/unet_partnet_randinit/Bottle/ratio_1.00 SOLVER.max_iter 3000 SOLVER.step_size 1500,750 SOLVER.test_every_iter 200 SOLVER.test_iter 84 SOLVER.ckpt '' DATA.train.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Bottle_train_level3.tfrecords DATA.train.take 315 DATA.train.batch_size 8 DATA.test.location /media/christina/Data/ANFASS_data/partnet_data/dataset/Bottle_test_level3.tfrecords MODEL.nout 9 MODEL.factor 2 LOSS.num_class 9
