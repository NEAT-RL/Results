#!/usr/bin/env python
import pandas as pd
import numpy as np
import math
from scipy import stats
import scipy as sp

import csv
filenames = ['em_power_po_space-invaders.csv', 'em_power_po_variant_space-invaders.csv', 'em_power_po_gradient_space-invaders.csv', 'neat_space-invaders.csv', 'power_po_space-invaders.csv', 'power_po_variant_space-invaders.csv', 'power_po_gradient_space-invaders.csv']
groups = []
min_length = 10000
for filename in filenames:
    with open(filename, 'rU') as p:
        my_list = [list(map(float, rec)) for rec in csv.reader(p, delimiter=',')]
        group = my_list[-1:][0][1:]
        if len(group) < min_length:
            min_length = len(group)
        groups.append(group)
        print(group)

stripped_groups = []
for group in groups:
    stripped_groups.append(group[:min_length])

f, p = stats.f_oneway(stripped_groups[0], stripped_groups[1], stripped_groups[2], stripped_groups[3], stripped_groups[4])

print('One-way ANOVA')
print('=============')

print('F value:', f)
print('P value:', p, '\n')


from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import itertools

headers = ['P-PG', 'P-TD', 'P-RMSPROP', 'NEAT', 'NEAT-EM-P-PG','NEAT-EM-P-TD', 'NEAT-EM-P-RMSPROP']
group_names = []
for header in headers:
    group_names += list(itertools.repeat(header, min_length))

mc = MultiComparison(np.asarray(stripped_groups).flatten(), group_names)
result = mc.tukeyhsd()

from statsmodels.stats.libqsturng import psturng

print(result)
print(mc.groupsunique)
print(psturng(np.abs(result.meandiffs / result.std_pairs), len(result.groupsunique), result.df_total))


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return h


count = 0
for stripped_group in stripped_groups:
    print(headers[count])
    print(np.mean(stripped_group), mean_confidence_interval(stripped_group))
    count+=1


