#!/usr/bin/env python
import pandas as pd
import glob

import seaborn as sns
import numpy as np
import math
from  matplotlib.ticker import FuncFormatter
from matplotlib import  pyplot as plt


import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return h

"""Description:
This is a plot program created by yiming for SEAL 2017
"""
problem_name = 'mc'
experiment_type = 'effectiveness'
graph_type = 'line'
allFiles = glob.glob("results/sum/" + graph_type + '/' +
                     experiment_type
                     + "/" + problem_name +
                     "/*.csv")
allFiles = ['neat_space-invaders.csv', 'power_po_gradient_space-invaders.csv', 'power_po_variant_space-invaders.csv', 'power_po_space-invaders.csv', 'em_power_po_space-invaders.csv']
flatui = ["#9b59b6", "#3498db",  "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
experiments = ['NEAT', 'NEAT-EM-P-RMSPROP', 'NEAT-EM-P-TD', 'NEAT-EM-P-PG', 'EM-PG']

experiments2 = ['P-PG', 'P-RMSPROP', 'P-TD', 'NEAT-EM-P-PG', 'NEAT-EM-P-RMSPROP', 'NEAT-EM-P-TD']
#experiments = ['RAC with Evolved NN based Features', 'RAC with '
#                                                    'Discretization based
# Features']

sns.set(font_scale = 1.5)
count = 0
for file_ in allFiles:
    # df = pd.read_csv(file_, index_col=None, header=0, names=[str(count)])
    #df = pd.read_csv(file_, index_col=0).rolling(window=2).mean()
    df = pd.read_csv(file_, index_col=0)
    #print(df.T.mean())
    # h = mean_confidence_interval(df.T.values)
    # ci_list = []
    # for e in h:
    #     if not math.isnan(e):
    #         ci_list.append(e)
    #     else:
    #         ci_list.append(0.0)

    ax = sns.tsplot(data=df.T.values, color=flatui[count],
                    condition=experiments[count], err_style="ci_band",
                    ci=[95])
    # ax = sns.tsplot(data=df.T.values[:,1:101], ci="sd")
    count += 1

# frame = pd.concat(list_, axis=0)
#mean = frame.mean(axis=1)
#print()
# stderror = frame.std(axis=1)/30

# frame.to_csv("test.csv")
#ax.set(xlabel='Learning Episodes', ylabel='Average Steps')
# ax.set_xlabel('Learning Episodes', fontsize=18)
# ax.set_ylabel('Average Steps', fontsize=18)
# ax.set(xlabel='Generations', ylabel='Average Steps')
ax.set_xlabel('Iterations/Generations', fontsize=12)
ax.set_ylabel('Average Reward', fontsize=12)
sns.plt.title('Average Reward on SpaceInvaders-ram-v0')
# sns.plt.title('Average Reward Per Learning Episode on space-invaders-ram-v0')
# sns.plt.title('Average Steps to Reach Goal Region Per Generation')
# sns.plt.xticks(ax.get_xticks(), ax.get_xticks() * 5)
# sns.plt.set_major_locator(MaxNLocator(integer=True))
# sns.plt.xticks([0, 2000, 4000, 6000, 8000, 10000])
#sns.plt.yticks([0, 20, 40, 60, 80, 100])
# sns.plt.xlim(0, 10000)
# sns.plt.xlim(0, 200)
#sns.plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y/10)))

#
# legend = sns.plt.legend(frameon = 1)
# frame = legend.get_frame()
# frame.set_facecolor('gray')
# frame.set_edgecolor('red')

# sns.plt.savefig('mc-steps.pdf', format='pdf')
#sns.plt.savefig('rac-mcp-steps.pdf', format='pdf')
#sns.plt.savefig('rac-mcp-steps.pdf', format='pdf')
plt.savefig('space-invaders_presentation.png', format='png')
#sns.plt.savefig('mc-steps.pdf', format='pdf')
sns.plt.show()

