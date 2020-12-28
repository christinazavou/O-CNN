import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def read_overall_file():
    # read all experiments overall stats
    stats = open(OVERALL, newline="").readlines()
    # get level of computation on mesh (points/faces/components)
    prefix = stats.pop(0).strip().split(",")
    prefix = [p for p in prefix if p != ""]
    prefix.pop(0)

    # get stat names/description
    metrics = stats.pop(0).strip().split(",")[2:]
    metrics = list(dict.fromkeys(metrics))

    # get experiment name/description and stats
    exp_stats = []
    exp_name = []
    for line in stats:
        line = line.split(",")
        if line[0] != "":
            exp_name.append(line[0][13:])
        exp_stats.append([float(j) for j in line[2:]])

    exp_stats = np.array(exp_stats)
    best_exp_stats = []

    # get highest scores per experiment
    for i in range(len(exp_name)):
        best_exp_stats.append(np.amax(exp_stats[i * 3:i * 3 + 3], axis=0))
    best_exp_stats = np.array(best_exp_stats)

    # create experiment graph
    create_overall_graphs(metrics, exp_name, best_exp_stats, prefix)


def read_per_label_file():
    # read all experiments per label stats
    stats = open(PER_LABEL, newline="").readlines()
    # get level of computation for labels
    prefix = stats.pop(0).strip().split(",")
    prefix = [p for p in prefix if p != ""]
    prefix.pop(0)

    # get experiment name/description and stats
    exp_stats = []
    exp_name = []
    for line in stats:
        line = line.split(",")
        if line[0] != "":
            exp_name.append(line[0])
        exp_stats.append([float(j) for j in line[2:]])
    exp_stats = np.array(exp_stats)

    # create experiments per experiment per label
    create_per_label_graphs(exp_name, exp_stats, prefix)


def create_per_label_graphs(exp_name, exp_stats, prefix):
    # decide colour and style of line graph
    def graph_style(idx):
        switcher = {
            0: "r-o",
            1: "g--^",
            2: "b:s"
        }
        return switcher.get(idx, "Error")

    N = np.arange(len(prefix))
    for i in range(len(exp_name)):
        fig, ax = plt.subplots(figsize=[15, 10])
        ax.grid(axis='both', which='both', color='black',
                linestyle='-.', linewidth=0.5)
        plt.title(exp_name[i] + "\nPer Label Metrics")
        ax.set_xticks(N)
        plt.xticks(rotation=90)
        ax.set_yticks(np.arange(0, 101, 5.0))
        ax.set_xticklabels(prefix)
        for j in range(3):
            plt.plot(N, exp_stats[i * 3 + j], graph_style(j))
        ax.legend(["points", "faces", "components"])
        plt.savefig(fname=os.path.join(OUT_DIR, '{}_per_label.png'.format(exp_name[i])), bbox_inches='tight')


def create_overall_graphs(metrics, exp_name, best_exp_stats, prefix):
    N = len(exp_name)
    bar_size = 0.25
    bar_idx = np.arange(N)
    for k in range(len(prefix)):
        fig, ax = plt.subplots(figsize=[10, 10])
        ax.set_yticks(bar_idx + bar_size * (len(metrics) - 1) / 2)
        ax.set_yticklabels(exp_name)
        ax.grid(axis='both', which='both', color='black',
                linestyle='-.', linewidth=0.5)
        for i in range(len(metrics)):
            ax.barh(bar_idx, best_exp_stats[:, k * len(metrics) + i], height=bar_size - 0.1, label=metrics[i])
            for j, v in enumerate(best_exp_stats[:, k * len(metrics) + i]):
                ax.text(v + 1, bar_idx[j] - 0.05, str(v), color='black', size='small')
            bar_idx = bar_idx + bar_size

        ax.set_xticks(np.arange(0, 101, 10.0))
        plt.title("Per " + prefix[k] + "Overall Metrics")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(fname=os.path.join(OUT_DIR, 'over_all_results_per_{}.png'.format(prefix[k])), bbox_inches='tight')


OVERALL = "/home/maria/Downloads/O-HRNet experiments - Results.csv"
assert (os.path.exists(OVERALL))

PER_LABEL = "/home/maria/Downloads/O-HRNet experiments - Per Label IoU.csv"
assert (os.path.exists(PER_LABEL))

OUT_DIR = "./"
if len(sys.argv) > 1:
    print(sys.argv)
    OUT_DIR = sys.argv[1]
os.makedirs(OUT_DIR, exist_ok=True)

read_overall_file()
read_per_label_file()
