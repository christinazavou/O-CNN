import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def vis_confusion_matrix(matrix, categories, colors):
    df_cm = pd.DataFrame(matrix, index=categories, columns=categories)
    plt.figure()
    g = sn.heatmap(df_cm, annot=True, fmt='.1f', xticklabels=categories, yticklabels=categories)
    g.set_xticklabels(categories, rotation=-65)

    for cat_idx in range(len(categories)):
        g.get_xticklabels()[cat_idx].set_color('#{:06x}'.format(colors[cat_idx]))
        g.get_yticklabels()[cat_idx].set_color('#{:06x}'.format(colors[cat_idx]))

    plt.show()


# vis_confusion_matrix([[1,1,1],[2,2,2],[3,3,3]], ['a', 'b', 'c'], [14621210, 2960781, 2468886])