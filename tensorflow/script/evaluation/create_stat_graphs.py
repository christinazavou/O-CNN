import matplotlib.pyplot as plt
import numpy as np
import os

OVERALL = "/home/maria/Downloads/O-HRNet experiments - Results.csv"
assert (os.path.exists(OVERALL))

PER_LABEL = "/home/maria/Downloads/O-HRNet experiments - Per Label IoU.csv"
assert (os.path.exists(PER_LABEL))

cont = open(OVERALL, newline="").readlines()
prefix = cont.pop(0).strip().split(",")
prefix = [p for p in prefix if p != ""]
prefix.pop(0)

metrics = cont.pop(0).strip().split(",")[2:]
metrics = list(dict.fromkeys(metrics))

exp_stats = []
exp_name = []
for line in cont:
    print(line)
    line = line.split(",")
    if line[0] != "":
        exp_name.append(line[0][13:])
    exp_stats.append([float(j) for j in line[2:]])

exp_stats = np.array(exp_stats)
best_exp_stats = []
print(exp_name)
for i in range(len(exp_name)):
    best_exp_stats.append(np.amax(exp_stats[i * 3:i * 3 + 3], axis=0))
best_exp_stats = np.array(best_exp_stats)
print(best_exp_stats)

fig, ax = plt.subplots(figsize=[10, 10])
N = len(exp_name)
bar_size = 0.25
bar_idx = np.arange(N)
ax.set_yticks(bar_idx+bar_size*(len(metrics)-1)/2)
ax.set_yticklabels(exp_name)

ax.grid(axis='both', which='both', color='black',
        linestyle='-.', linewidth=0.5)
for i in range(len(metrics)):
    ax.barh(bar_idx, best_exp_stats[:, i], height=bar_size - 0.1,label=metrics[i])
    for j, v in enumerate(best_exp_stats[:,i]):
        ax.text(v + 1, bar_idx[j]-0.05, str(v), color='black',size='small')
    bar_idx = bar_idx + bar_size

ax.set_xticks(np.arange(0, 101, 10.0))
plt.title(prefix[0]+" Metrics")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.show()

plt.savefig(fname='./test.png', bbox_inches='tight')
