#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HEAR Leaderboard - Results Postprocessing

For these summary figures, we normalize each model/task score.
Normalized scores allow us to compare models and tasks against each
other, under the assumption each task is equally weighted.
Normalized scores are used to show the heat-value of each model on
each HEAR task.

The normalization procedure is as follows:

1) For each task, we standardize the scores to zero mean and unit
variance. Unlike transforming tasks to ranks, we assume that the
scale of intra-task scores is important.

2) The standardized scores are Winsorized (clamped) to have variance
within [-1, +1]. By limiting the importance of extremely high or
low scores on a single task, this approach allows for better
inter-task comparison.

For correlation tables, only the highest and lowest correlations
are displayed.

Cells are sorted to minimize the traveling salesperson distance.
"""



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import copy
import copy
from py2opt.routefinder import RouteFinder


from tqdm.auto import tqdm
from py2opt.routefinder import RouteFinder

import os

output = 'results.csv'
os.system("wget -O leaderboard.csv https://raw.githubusercontent.com/hearbenchmark/hear-benchmark/main/docs/leaderboard.csv")
df_new = pd.read_csv("leaderboard.csv")
df = df_new.set_index('Model').drop(columns=['URL'])
df

"""## Impute values for dimensionality reduction"""


X_full = df.copy().to_numpy()
imp = IterativeImputer(max_iter=10, random_state=0)
X_full_imp = imp.fit_transform(X_full)

df_imputed = df.copy()
df_imputed[:] = X_full_imp

"""### Sort leaderboard by avg scores"""

df_imputed_avg = df_imputed.copy()
df_imputed_avg["avgscore"] = df_imputed_avg.mean(axis=1)
df_imputed_avg = df_imputed_avg.sort_values("avgscore", ascending=False)


# Separating out the features
xorig = df_imputed.loc[:, :].values
# Standardizing the features
x = StandardScaler().fit_transform(xorig)

df_imputed_scaleavg = df_imputed.copy()
df_imputed_scaleavg[:] = x

# Winsorize (censor) everyone above 1 std in the scaled scores
df_imputed_scaleavg_winsor = df_imputed_scaleavg.clip(-1, +1)

df_imputed_scaleavg["avgscore"] = df_imputed_scaleavg.mean(axis=1)
df_imputed_scaleavg = df_imputed_scaleavg.sort_values("avgscore", ascending=False)


df_imputed_scaleavg_winsor["avgscore"] = df_imputed_scaleavg_winsor.mean(axis=1)
df_imputed_scaleavg_winsor = df_imputed_scaleavg_winsor.sort_values("avgscore", ascending=False)
df_imputed_scaleavg_winsor

#plt.plot(sorted(df_imputed_scaleavg.values.flatten()))
#plt.plot(sorted(df_imputed_scaleavg_winsor.values.flatten()))



# Separating out the features
xorig = df_imputed.loc[:, :].values
# Standardizing the features
x = QuantileTransformer().fit_transform(xorig)

df_imputed_rank = df_imputed.copy()
df_imputed_rank[:] = x
df_imputed_rank["avgscore"] = df_imputed_rank.mean(axis=1)
df_imputed_rank = df_imputed_rank.sort_values("avgscore", ascending=False)
#display(df_imputed_rank)

for x in list(zip(
    list(df_imputed_avg.sort_values("avgscore", ascending=False).index),
    list(df_imputed_scaleavg.sort_values("avgscore", ascending=False).index),
    list(df_imputed_scaleavg_winsor.sort_values("avgscore", ascending=False).index),
    list(df_imputed_rank.sort_values("avgscore", ascending=False).index))):
  print(x)

# TBH scaling then meaning seems the most reasonable prior to me

#df["model"] == list(df_imputed_scaleavg_winsor.index)
df = df.loc[list(df_imputed_scaleavg_winsor.index)]
df_imputed = df_imputed.loc[list(df_imputed_scaleavg_winsor.index)]

df

df_with_grand_score = df.copy()
df_with_grand_score.insert(0, "Grand Score", df_imputed_scaleavg_winsor["avgscore"])
df_with_grand_score.insert(0, "URL", df_new["URL"])
df_with_grand_score.to_csv("results/leaderboard-with-grand-score.csv", index=False)
df_with_grand_score

df_imputed_scaleavg_winsor = df_imputed_scaleavg_winsor.drop("avgscore", axis=1)

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.viridis.html
#CMAP = "viridis"
#CMAP = "plasma"
CMAP = "inferno"

# Middle value of palette for blank values
middle_value = plt.get_cmap(CMAP)(0.5)
print(middle_value)
# Swap red and blue so NaNs stand out a little
middle_value = (middle_value[2], middle_value[1], middle_value[0], middle_value[3])

"""The scaled, winsored scores are what we use for most of our analysis"""

df_scaleavg_winsor = df_imputed_scaleavg_winsor.copy()
df_scaleavg_winsor[df.isnull()]=np.nan
df_scaleavg_winsor

def zeroannot(df, sigfig=3):
  df_annot = df.astype('float')
  if sigfig == 3:
    df_annot = df_annot.applymap(lambda x: f"{x:.3f}")
  elif sigfig == 2:
    df_annot = df_annot.applymap(lambda x: f"{x:.2f}")
  else:
    assert False
  df_annot = df_annot.applymap(lambda x: x.replace("0.", "."))
  return df_annot



fig, ax = plt.subplots(figsize=(12,15))         # Sample figsize in inches
#g = sns.heatmap(df_scaleavg_winsor, annot=df.astype('float'), ax=ax, fmt='.3f',
g = sns.heatmap(df_scaleavg_winsor, annot=zeroannot(df), ax=ax, fmt='', annot_kws={"size": 12},
                cmap=CMAP, cbar=False,
                linewidths=0.1)
g.set_facecolor(middle_value)
ax.tick_params(right=False, top=True, labelright=False, labeltop=True)
#Rotate X ticks
plt.xticks(rotation='vertical')
g.figure.savefig("results/leaderboard.eps",bbox_inches='tight')
g.figure.savefig("results/leaderboard.png",bbox_inches='tight')

SIGFIG = 3


"""## Correlations

"""



"""
### Model vs Model

Using results without imputed values"""

df_for_corr = df.copy()
df_for_corr = df_scaleavg_winsor.copy()

import numpy.ma as ma

df_corr_models = pd.DataFrame(index=df_for_corr.index, columns=df_for_corr.index)

for i, row_x in df_for_corr.iterrows():

  for j, row_y in df_for_corr.iterrows():

    list_x = row_x.values
    list_y = row_y.values

    list_x_nanmask=ma.masked_invalid(list_x)
    list_y_nanmask=ma.masked_invalid(list_y)

    msk = (~list_x_nanmask.mask & ~list_y_nanmask.mask)

    corr = ma.corrcoef(list_x[msk], list_y[msk])
    corr = corr[0][1]
    corr = round(corr, SIGFIG)
    df_corr_models.at[i, j] = float(corr)

df_corr_models

"""### Task vs Task

Using results without imputed values
"""

df_corr_tasks = pd.DataFrame(index=df_for_corr.columns, columns=df_for_corr.columns)

for i, row_x in df_for_corr.T.iterrows():

  for j, row_y in df_for_corr.T.iterrows():

    list_x = row_x.values
    list_y = row_y.values

    list_x_nanmask=ma.masked_invalid(list_x)
    list_y_nanmask=ma.masked_invalid(list_y)

    msk = (~list_x_nanmask.mask & ~list_y_nanmask.mask)

    corr = ma.corrcoef(list_x[msk], list_y[msk])
    corr = corr[0][1]
    corr = round(corr, SIGFIG)
    df_corr_tasks.at[i, j] = float(corr)

df_corr_tasks

# TBH both these libraries give similar approximations on our matrices, but py2opt
# is slightly better on task v task
#!pip3 install python-tsp
#!pip3 install py2opt
# https://github.com/pdrm83/py2opt/pull/8
#!pip3 install git+https://github.com/turian/py2opt@verbose

plt.plot(sorted(df_corr_models.to_numpy().flatten()))
plt.show()
plt.plot(sorted(df_corr_tasks.to_numpy().flatten()))
plt.show()

MODEL_CORR_PERCENTILE = 15
TASK_CORR_PERCENTILE = 20
model_corr_min = np.percentile(df_corr_models.to_numpy().flatten(), MODEL_CORR_PERCENTILE)
model_corr_max = np.percentile(df_corr_models.to_numpy().flatten(), 100 - MODEL_CORR_PERCENTILE)
task_corr_min = np.percentile(df_corr_tasks.to_numpy().flatten(), TASK_CORR_PERCENTILE)
task_corr_max = np.percentile(df_corr_tasks.to_numpy().flatten(), 100 - TASK_CORR_PERCENTILE)
print(model_corr_min, model_corr_max)
print(task_corr_min, task_corr_max)

# Convert correlations to distances
df_corr_models_dist = (df_corr_models.to_numpy() * (-1) + 1) / 2
df_corr_tasks_dist = (df_corr_tasks.to_numpy() * (-1) + 1) / 2

# Compare the two TSP heuristics
"""
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing
from py2opt.routefinder import RouteFinder

distance_matrix = df_corr_models_dist
distance_matrix[:, 0] = 0

distances = []

permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
print(distance, permutation)
distances.append(distance)

permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
print(distance, permutation)
distances.append(distance)

for i in tqdm(range(3)):
  route_finder = RouteFinder(df_corr_models_dist, list(df_corr_models.index), iterations=50)
  distance, permutation = route_finder.solve()
  print(distance, permutation)
  distances.append(distance)

print()
for d in distances:
  print(d)
"""


def rotate(l, n):
    return l[-n:] + l[:-n]

orig_df_corr_models_dist = copy.deepcopy(df_corr_models_dist)
orig_df_corr_models = copy.deepcopy(df_corr_models)

model_distance_permutations = []
for r in tqdm(list(range(df_corr_models_dist.shape[0]))):
  rotidx = rotate(list(range(orig_df_corr_models_dist.shape[0])),r)
  df_corr_models_dist = copy.deepcopy(orig_df_corr_models_dist[rotidx,:][:,rotidx])
  df_corr_models = copy.deepcopy(orig_df_corr_models.iloc[rotidx])
  print(list(df_corr_models.index))
  for i in tqdm(range(10)):
    route_finder = RouteFinder(df_corr_models_dist, list(df_corr_models.index), iterations=250, verbose=False)
    distance, permutation = route_finder.solve()
    model_distance_permutations.append((distance, permutation))
  model_distance_permutations.sort()
  distance, permutation = model_distance_permutations[0]
  print(distance, permutation)

model_distance_permutations.sort()
distance, permutation = model_distance_permutations[0]
print(distance, permutation)

# Try different starting points to find best task to start from for TSP

def rotate(l, n):
    return l[-n:] + l[:-n]


orig_df_corr_tasks_dist = copy.deepcopy(df_corr_tasks_dist)
orig_df_corr_tasks = copy.deepcopy(df_corr_tasks)


task_distance_permutations = []
for r in tqdm(list(range(df_corr_tasks_dist.shape[0]))):
  rotidx = rotate(list(range(orig_df_corr_tasks_dist.shape[0])),r)
  df_corr_tasks_dist = copy.deepcopy(orig_df_corr_tasks_dist[rotidx,:][:,rotidx])
  df_corr_tasks = copy.deepcopy(orig_df_corr_tasks.iloc[rotidx])
  print(list(df_corr_tasks.index))
  for i in tqdm(range(10)):
    route_finder = RouteFinder(df_corr_tasks_dist, list(df_corr_tasks.index), iterations=250, verbose=False)
    distance, permutation = route_finder.solve()
    task_distance_permutations.append((distance, permutation))
  task_distance_permutations.sort()
  distance, permutation = task_distance_permutations[0]
  print(distance, permutation)

task_distance_permutations.sort()
distance, permutation = task_distance_permutations[0]
print(distance, permutation)

"""
from py2opt.routefinder import RouteFinder
from tqdm.auto import tqdm

distance_permutations = []
for i in tqdm(range(3)):
  route_finder = RouteFinder(df_corr_models_dist, list(df_corr_models.index), iterations=500)
  distance, permutation = route_finder.solve()
  distance_permutations.append((distance, permutation))
"""

model_distance_permutations.sort()
distance, permutation = model_distance_permutations[0]

print(distance, permutation)

# Sort the matrix by this
df_corr_models = df_corr_models.loc[permutation][permutation]

# Let's show only the extreme values
df_corr_models = df_corr_models.astype('float')
df_corr_models_winsor = df_corr_models.copy()
print(model_corr_max, model_corr_min)
#df_corr_models_winsor = df_corr_models_winsor.where(((df_corr_models > model_corr_max) | (df_corr_models <= model_corr_min)) & (df_corr_models != 1))
df_corr_models_winsor = df_corr_models_winsor.where(((df_corr_models > model_corr_max) | (df_corr_models <= model_corr_min)))
np.fill_diagonal(df_corr_models_winsor.to_numpy(), np.nan)

fig, ax = plt.subplots(figsize=(15*0.97,17*0.97))         # Sample figsize in inches
#sns.heatmap(df_corr_models_winsor, annot=True, ax=ax, fmt='.2f', cmap=CMAP, cbar=False)
sns.heatmap(df_corr_models_winsor, annot=zeroannot(df_corr_models_winsor, sigfig=2), annot_kws={"size": 12}, ax=ax, fmt='', cmap=CMAP, cbar=False)
ax.tick_params(right=False, top=True, labelright=False, labeltop=True)
#Rotate X ticks
plt.xticks(rotation='vertical')
plt.savefig("results/model_correlation.eps",bbox_inches='tight')
plt.savefig("results/model_correlation.png",bbox_inches='tight')

"""
from py2opt.routefinder import RouteFinder

distance_permutations = []
for i in tqdm(range(3)):
  route_finder = RouteFinder(df_corr_tasks_dist, list(df_corr_tasks.index), iterations=500)
  distance, permutation = route_finder.solve()
  distance_permutations.append((distance, permutation))
"""

task_distance_permutations.sort()
distance, permutation = task_distance_permutations[0]
print(distance, permutation)

# Sort the matrix by this
df_corr_tasks = df_corr_tasks.loc[permutation][permutation]

# Let's show only the extreme values
df_corr_tasks = df_corr_tasks.astype('float')
df_corr_tasks_winsor = df_corr_tasks.copy()
#df_corr_tasks_winsor = df_corr_tasks_winsor.where(((df_corr_tasks > task_corr_max) | (df_corr_tasks <= task_corr_min)) & (df_corr_tasks != 1))
df_corr_tasks_winsor = df_corr_tasks_winsor.where(((df_corr_tasks > task_corr_max) | (df_corr_tasks <= task_corr_min)))
np.fill_diagonal(df_corr_tasks_winsor.to_numpy(), np.nan)

fig, ax = plt.subplots(figsize=(12*0.80,14*0.80))         # Sample figsize in inches
#sns.heatmap(df_corr_tasks_winsor.astype('float'), annot=True, ax=ax, fmt='.2f', cmap=CMAP, cbar=False, square=True)
sns.heatmap(df_corr_tasks_winsor.astype('float'), annot=zeroannot(df_corr_tasks_winsor, sigfig=2), annot_kws={"size": 11}, ax=ax, fmt='', cmap=CMAP, cbar=False, square=True)
ax.tick_params(right=False, top=True, labelright=False, labeltop=True)
#Rotate X ticks
plt.xticks(rotation='vertical')
plt.savefig("results/task_correlation.eps",bbox_inches='tight')
plt.savefig("results/task_correlation.png",bbox_inches='tight')

#df_corr_tasks.mean()
#df_corr_tasks.apply(pd.DataFrame.describe, axis=1).sort_values("mean")
