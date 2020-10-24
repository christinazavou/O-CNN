import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn


def vis_confusion_matrix(matrix, categories, colors, title):
    pd_data = pd.DataFrame(matrix)
    sum_rows = pd_data.sum(axis=1).values.reshape((matrix.shape[0], 1))
    percentages = np.divide(matrix, sum_rows, out=np.zeros_like(matrix), where=sum_rows != 0).astype(
        np.int) * 100
    pd_data = pd.DataFrame(percentages)
    annotations = [["{}%".format(c) for c in row] for row in pd_data.values]
    pd_ann = pd.DataFrame(annotations)
    plt.figure(figsize = [20,20])
    ylabels = ["{} ({})".format(cat, sum_r[0]) for cat, sum_r in zip(categories, sum_rows)]
    g = sn.heatmap(pd_data, annot=pd_ann, square=True,fmt='', xticklabels=categories, yticklabels=ylabels)
    g.set_xticklabels(categories, rotation=-65)
    g.set_xlabel('Prediction')
    g.set_ylabel('Actual')

    for cat_idx in range(len(categories)):
        c = '#{:06x}'.format(colors[cat_idx]) if isinstance(colors[cat_idx], int) else colors[cat_idx]
        g.get_xticklabels()[cat_idx].set_color(c)
        g.get_yticklabels()[cat_idx].set_color(c)
    plt.title(title)
    plt.savefig("/media/maria/BigData1/Maria/repos/ocnn_logs/seg/hrnet_no_colour_12_rot_depth7_100epochs_weighted_plateau_sgd/confusion_matrix.png",bbox_inches = 'tight')
    #plt.show()

# vis_confusion_matrix(np.array([[1,1,1],[0,0,0],[3,3,3]]), ['a', 'b', 'c'], [14621210, 2960781, 2468886], "ena")
# vis_confusion_matrix(np.array([[1,1,1],[0,0,0],[3,3,3]]), ['a', 'b', 'c'], ["#ff9180", "#bf3069", "#bf3069"], "ena")
