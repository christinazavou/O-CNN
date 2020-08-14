SHELL := /bin/bash # otherwise we can't use "source"

#----------------------------------------------------------------------------------------------------------------------#
loc="/media/christina/Data/ANFASS_data/O-CNN/ocnn_completion"
utils="/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/util"

generate_cases:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${utils} \
		&& python convert_tfrecords.py --file_dir ${loc}/shape.points --list_file ${loc}/specific_test_pointlist.txt --records_name ${loc}/specific_test_points.tfrecords \
		&& python convert_tfrecords.py --file_dir ${loc}/shape.octrees --list_file ${loc}/specific_test_octreelist.txt --records_name ${loc}/specific_test_octrees.tfrecords
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
scripts="/home/christina/Documents/ANNFASS_code/zavou-repos/O-CNN/tensorflow/script"

predict-resnet:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${scripts} \
		&& python run_ae.py --config configs/ae_resnet_decode.yaml

predict-ocnn:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
		&& source activate OCNN \
		&& cd ${scripts} \
		&& python run_ae.py --config configs/ae_ocnn_decode.yaml
#----------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------#
predict: predict-ocnn predict-resnet

model_prefix="_b16_decode_shape"
logs_path="/media/christina/Data/ANFASS_data/O-CNN/output/ocnn_completion/ae"

visualize-resnet:
	chmod 777 visualize_generated_octrees.sh
	./visualize_generated_octrees.sh ${logs_path}/resnet${model_prefix}

visualize-ocnn:
	chmod 777 visualize_generated_octrees.sh
	./visualize_generated_octrees.sh ${logs_path}/ocnn${model_prefix}

visualize-ocnn: visualize-resnet visualize-ocnn
#----------------------------------------------------------------------------------------------------------------------#
