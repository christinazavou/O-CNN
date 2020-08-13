SHELL := /bin/bash # otherwise we can't use "source"

loc="/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion"
utils="/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/util"
scripts="/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script"

generate_cases:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${utils} \
		&& python convert_tfrecords.py --file_dir ${loc}/shape.points --list_file ${loc}/specific_test_pointlist.txt --records_name ${loc}/specific_test_points.tfrecords \
		&& python convert_tfrecords.py --file_dir ${loc}/shape.octrees --list_file ${loc}/specific_test_octreelist.txt --records_name ${loc}/specific_test_octrees.tfrecords

predict:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${scripts} \
		&& python run_ae.py --config configs/ae_resnet_decode.yaml \
		&& python run_ae.py --config configs/ae_ocnn_decode.yaml

visualize:
	chmod 777 visualize_generated_octrees.sh
	./visualize_generated_octrees.sh ${scripts}/logs/ae/resnet_b16_decode_shape
	./visualize_generated_octrees.sh ${scripts}/logs/ae/ocnn_b16_decode_shape
