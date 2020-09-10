import numpy as np
import pickle

labels_original = np.loadtxt('/media/christina/Elements/ANNFASS_DATA/RGBA_uniform/with_colour/w_colour_norm_w_labels/RESIDENTIALvilla_mesh5826_w_label.txt')
labels_original = labels_original[:, -1].reshape(-1)

labels_final_points = pickle.load(open('/media/christina/Elements/ANNFASS_DATA/output/segmentation/Building/hrnet_randinit/Batch4NoColourDepth6/predicted_pkl_Building/p100000_m99909_i0.pkl', 'rb'))
labels_final_points = np.array(labels_final_points['orig']).reshape(-1)

assert np.array_equal(labels_original, labels_final_points)
