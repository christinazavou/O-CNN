SHELL := /bin/bash # otherwise we can't use "source"

root_dir_sample = /media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour_sample
root_dir = /media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour

make-buildings-no-colour-sample:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
	&& source activate OCNN \
	&& cd ../data \
	&& python seg_buildings_no_colour.py """from_colored_annotated_data_to_default_ply('${root_dir_sample}', 'w_colour_norm_w_labels', sample_pts=1000, parallel=True)""" \
	&& python seg_buildings_no_colour.py """convert_points('${root_dir_sample}', sample_pts=1000, parallel=False)""" \
	&& python seg_buildings_no_colour.py """convert_points_to_tfrecords('${root_dir_sample}', 'dataset_points_sample', sample_pts=1000)"""

make-buildings-no-colour:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
	&& source activate OCNN \
	&& cd ../data \
	&& python seg_buildings_no_colour.py """from_colored_annotated_data_to_default_ply('${root_dir}', 'w_colour_norm_w_labels', sample_pts=None, parallel=True)""" \
	&& python seg_buildings_no_colour.py """convert_points('${root_dir}', sample_pts=None, parallel=False)""" \
	&& python seg_buildings_no_colour.py """convert_points_to_tfrecords('${root_dir}', 'dataset_points_tuesday', sample_pts=None)"""

make-builgings-with-colour:
	export PATH=/home/christina/miniconda3/bin/:$$PATH \
	&& source activate OCNN \
	&& cd ../data \
	&& python seg_buildings_with_colour.py --file_dir '${root_dir}/w_colour_norm_w_labels' \
		--list_file '${root_dir}/dataset_points_sample/test.txt' \
		--records_name '${root_dir}/dataset_points_sample/test.tfrecords' \
	&& 	&& python seg_buildings_with_colour.py --file_dir '${root_dir}/w_colour_norm_w_labels' \
		--list_file '${root_dir}/dataset_points_sample/train.txt' \
		--records_name '${root_dir}/dataset_points_sample/train.tfrecords' \
		--shuffle_data
