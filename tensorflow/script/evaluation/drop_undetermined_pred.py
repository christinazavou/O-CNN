import os
import numpy as np

IN_DIR = "/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/cluster_exp_results/Seg_2k_hrnet_no_colour_depth_5_weighted_sgd_3_0.0/best_iou/probabilities"
assert (os.path.exists(IN_DIR))

OUT_DIR = "/media/maria/BigData1/Maria/buildnet_data_2k/100K_inverted_normals/cluster_exp_results/Seg_2k_hrnet_no_colour_depth_5_weighted_sgd_3_0.0/best_iou/probabilities_no_und"

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

for root, _, files in os.walk(IN_DIR):
    for f in files:
        try:
            x = np.load(os.path.join(root, f))
        except IOError:
            print("Cannot read file {}. Moving to next.".format(f))
            continue
        except ValueError:
            print("Allowing pickle reading")
            x = np.load(os.path.join(root, f), allow_pickle=True)
        t = x[..., 1:]
        t = np.true_divide(t, np.sum(t, axis=1)[:, None])
        np.save(os.path.join(OUT_DIR, f), t)
