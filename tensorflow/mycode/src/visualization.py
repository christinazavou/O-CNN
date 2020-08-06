import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


# from src.config import CLASS_TO_LABEL


class Visualizer:
    _bar_width = 0.35

    @staticmethod
    def bar_plot(y_values1, y_values2, x_labels, label_1, label_2, ylabel=None, title=None):
        x = np.arange(len(x_labels))

        fig, ax = plt.subplots()
        bars_1 = ax.bar(x - Visualizer._bar_width / 2, y_values1, Visualizer._bar_width, label=label_1)
        bars_2 = ax.bar(x + Visualizer._bar_width / 2, y_values2, Visualizer._bar_width, label=label_2)

        Visualizer._common_annotation(ax, ylabel=ylabel, title=title, xticks=x, xticklabels=x_labels, legend=True)
        Visualizer._annotated_bars(ax, bars_1)
        Visualizer._annotated_bars(ax, bars_2)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def _annotated_bars(ax, bars):
        for bar in bars:
            ax.annotate(str(bar.get_height()), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    @staticmethod
    def _common_annotation(ax, xlabel=None, ylabel=None, title=None, xticks=None, xticklabels=None, legend=False):
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, rotation=65, ha="right")
        if legend:
            ax.legend()

    @staticmethod
    def confusion_matrix(matrix, categories):
        df_cm = pd.DataFrame(matrix, index=categories, columns=categories)
        plt.figure()
        g = sn.heatmap(df_cm, annot=True, fmt='.1f', xticklabels=categories, yticklabels=categories)
        g.set_xticklabels(categories, rotation=-65)
        plt.show()

# Visualizer.bar_plot([2, 2, 2], [4, 4, 4], ['a', 'b', 'c'], 'ena', 'dio')
# Visualizer.confusion_matrix(np.random.random((40, 40)), CLASS_TO_LABEL.keys())
