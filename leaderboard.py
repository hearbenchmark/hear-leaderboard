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
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import gdown
import os

output = 'results.csv'
os.system("wget -O leaderboard.csv https://raw.githubusercontent.com/hearbenchmark/hear-benchmark/main/docs/leaderboard.csv")
df_new = pd.read_csv("leaderboard.csv")
df_new

task_rename = {
 'beehive_states-v2-full': "Beehive",
 'beijing_opera-v1.0-hear2021-full': "Beijing Opera",
 'dcase2016_task2-hear2021-full': "DCASE 2016",
 'esc50-v2.0.0-full': "ESC-50",
 'fsd50k-v1.0-full': "FSD50k",
 'gunshot_triangulation-v1.0-full': "Gunshot",
 'libricount-v1.0.0-hear2021-full': "Libricount",
 'maestro-v3.0.0-5h': "Maestro 5h",
 'mridangam_stroke-v1.5-full': "Mridangam Stroke",
 'mridangam_tonic-v1.5-full': "Mridangam Tonic",
 'nsynth_pitch-v2.2.3-50h': "NSynth Pitch 50h",
 'nsynth_pitch-v2.2.3-5h': "NSynth Pitch 5h",
 'speech_commands-v0.0.2-5h': "Speech commands 5h",
 'speech_commands-v0.0.2-full': "Speech commands full",
 'tfds_crema_d-1.0.0-full': "CREMA-D",
 'tfds_gtzan_music_speech-1.0.0-full': "GTZAN Music/Speech",
 'tfds_gtzan-1.0.0-full': "GTZAN Genre",
 'vocal_imitation-v1.1.3-full': "Vocal Imitation",
 'vox_lingua_top10-hear2021-full': "VoxLingua107 top 10"
}

df_raw = df_raw.replace({"task": task_rename})

"""
model_rename = {
 'audio_dbert': "Stellenbosch LSL DBERT",
# 'audiomlp': "ID56-SSL MLP (audio)",
 'audiomlp': "IUT-CSE MLP (audio)",
 'avg_hubert_crepe': "NTU-GURA Avg Hubert+Crepe",
 'avg_hubert_wav2vec2': "NTU-GURA Avg Hubert+wav2vec2",
 'avg_xwc': "NTU-GURA Avg Hubert+w2v2+Crepe",
 'base': "CP-JKU PaSST base",
 'base2level': "CP-JKU PaSST 2-level",
 'base2levelmel': "CP-JKU PaSST 2-level+mel",
 'cat_hubert_wav2vec2': "NTU-GURA Cat Hubert+wav2vec2",
 'cat_wc': "NTU-GURA Cat wav2vec2+crepe",
 'cat_xwc': "NTU-GURA Cat Hubert+w2v2+Crepe",
 'efficient_latent': "RedRice/Xiaomi EfficientNet-B2",
 'embed': "UDONS ViT",
 'fusion_cat_xwc': "NTU-GURA Fusion Cat Hubert+w2v2+Crepe",
 'fusion_cat_xwc_time': "NTU-GURA Fusion Cat Hubert+w2v2+Crepe (time)",
 'fusion_hubert_xlarge': "NTU-GURA Fusion Hubert",
 'fusion_wav2vec2': "NTU-GURA Fusion wav2vec2",
 'hearline': "ibkuroyagi conformer",
 'hubert_xlarge': "NTU-GURA Hubert",
# 'kwmlp': "ID56-SSL MLP (keyword)",
 'kwmlp': "IUT-CSE MLP (keyword)",
 'my_model': "music+speech wav2vec",
 'openl3_hear': "OpenL3",
 'panns_hear': "CVSSP PANNS",
 'serab_byols': "Logitech AI SERAB BYOL-S",
 'torchcrepe': "Crepe",
 'wav2clip_hear': "Descript/MARL Wav2CLIP",
 'wav2vec2': "wav2vec2",
 'wav2vec2_ddsp': "AMAAI Lab wav2vec2+DDSP",
 'yamnet_hear': "YAMNet",
}
"""

model_rename = {
 'audio_dbert':         "Stellenbosch LSL DBERT",
 'audiomlp':            "IUT-CSE MLP (audio)",
 'avg_hubert_crepe':    "GURA Avg Hubert+CREPE",
 'avg_hubert_wav2vec2': "GURA Avg Hubert+wav2vec2",
 'avg_xwc':             "GURA Avg H+w+C",
 'base':                "CP-JKU PaSST base",
 'base2level':          "CP-JKU PaSST 2lvl",
 'base2levelmel':       "CP-JKU PaSST 2lvl+mel",
 'cat_hubert_wav2vec2': "GURA Cat Hubert+wav2vec2",
 'cat_wc':              "GURA Cat wav2vec2+CREPE",
 'cat_xwc':             "GURA Cat H+w+C",
 'efficient_latent':    "RedRice EfficientNet-B2",
 'embed':               "Sony UDONS ViT",
 'fusion_cat_xwc':      "GURA Fuse Cat H+w+C",
 'fusion_cat_xwc_time': "GURA Fuse Cat H+w+C (t)",
 'fusion_hubert_xlarge':"GURA Fuse Hubert",
 'fusion_wav2vec2':     "GURA Fuse wav2vec2",
 'hearline':            "Kuroyanagi hearline",
 'hubert_xlarge':       "GURA Hubert",
 'kwmlp':               "IUT-CSE MLP (keyword)",
 'my_model':            "AMAAI w2v2 music+spch",
 'openl3_hear':         "OpenL3",
 'panns_hear':          "CVSSP PANNS",
 'serab_byols':         "Logitech SERAB BYOL-S",
 'torchcrepe':          "CREPE",
 'wav2clip_hear':       "Descript/MARL Wav2CLIP",
 'wav2vec2':            "wav2vec2",
 'wav2vec2_ddsp':       "AMAAI Lab w2v2+DDSP",
 'yamnet_hear':         "Soundsensing YAMNet",
}

df_raw = df_raw.replace({"model": model_rename})

"""TODO: Here as a sanity check we can dump all the things with blank test score. (Or filter the df and use the "full column and row" pandas display method.)

## Create Model vs Task table
"""

df = df_raw.pivot('model', 'task', 'test_score')
df

# Use the new thing
df = df_new.set_index('Model').drop(columns=['URL'])
df

"""## Impute values for dimensionality reduction"""

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

X_full = df.copy().to_numpy()
imp = IterativeImputer(max_iter=10, random_state=0)
X_full_imp = imp.fit_transform(X_full)

df_imputed = df.copy()
df_imputed[:] = X_full_imp

"""
for i in range(10):
  dfsamp = df.copy().to_numpy()

  # Mask 10% of cells

  np.random.seed(i)
  # random boolean mask for which values will be changed
  mask = np.random.randint(0,2,size=dfsamp.shape).astype(np.bool)

  # use your mask to replace values in your input array
  dfsamp[mask] = np.nan

  imp = IterativeImputer(max_iter=10, random_state=0)
  dfsamp_imp = imp.fit_transform(dfsamp)

  mse = np.mean(np.square(np.nan_to_num(dfsamp_imp - X_full)))

np.random.seed(0)
"""





"""### Sort leaderboard by avg scores"""

df_imputed_avg = df_imputed.copy()
df_imputed_avg["avgscore"] = df_imputed_avg.mean(axis=1)
df_imputed_avg = df_imputed_avg.sort_values("avgscore", ascending=False)

from sklearn.preprocessing import StandardScaler

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

plt.plot(sorted(df_imputed_scaleavg.values.flatten()))
plt.plot(sorted(df_imputed_scaleavg_winsor.values.flatten()))



from sklearn.preprocessing import QuantileTransformer

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
    assert False;
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

"""
import numpy as np

SIGFIG = 3

# Make extra version of data table with primary score included in the task name
df_raw_new = df_raw.copy()
df_raw_new['task\_with\_primary\_score'] = df_raw_new["task"] + " (" + df_raw_new["primary_score"] + ")"  # .str.replace("test_", "")
df_latex = df_raw_new.pivot('model', 'task\_with\_primary\_score', 'test_score')

df_latex = df_latex.loc[list(df_imputed_scaleavg.index)]

# Boldface the top ranking cells
tobf = df_latex.round(SIGFIG).rank(numeric_only=False, na_option='top') >= 26.5

df_latex_txt = df_latex.round(SIGFIG)

df_latex_txt = df_latex_txt.astype(str)
for idx in df_latex_txt.index:
  for col in df_latex_txt.columns:
    if tobf.at[idx, col]:
      df_latex_txt.at[idx, col] = "{\\em %s}" % df_latex_txt.at[idx, col]

df_latex_txt.index = pd.Series(df_latex_txt.index).replace(r"_", "\_", regex=True)
#df_latex_txt.replace(np.nan, "", inplace=True)
df_latex_txt.replace("nan", "", inplace=True)

df_latex_out = df_latex_txt.to_latex(escape=False,header=['\\rotatebox{90}{' + "\_".join(c.split("_")) + '}' for c in df.columns])
df_latex_out = "\\resizebox{\\textwidth}{!}{%\n" + df_latex_out + "\n}"

with open('results/model-v-task.tex', 'wt') as tf:
     tf.write(df_latex_out)

df_latex_txt
"""



"""## Best scores for each task, and each model"""





for task in df.columns:
  idxs = df_imputed[task].sort_values(ascending=False).index
  #df.iloc[idxs]

  fig, ax = plt.subplots(figsize=(2,16))         # Sample figsize in inches
  g = sns.heatmap(df_imputed_scaleavg.loc[idxs][[task]], annot=df.astype('float').loc[idxs][[task]], ax=ax, fmt='.3f', cmap=CMAP, cbar=False)
  g.set_facecolor(middle_value)
  ax.xaxis.tick_top() # x axis on top
  ax.xaxis.set_label_position('top')
  plt.show()

"""## Dimensionality reduction

### PCA

TODO (super low priority): could try other scalers
"""

# to help with plotting labels and not overlap
#!pip install adjustText

# ref https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text
from matplotlib.ticker import MaxNLocator

# Separating out the features
#xorig = df_imputed.loc[:, :].values
xorig = df_imputed_scaleavg_winsor.loc[:, :].values
# Standardizing the features
x = StandardScaler().fit_transform(xorig)

model_names = df_imputed.index.tolist()
task_names = df_imputed.T.index.tolist()

# Scaler matrix, models x tasks
x.shape

pca = PCA(n_components=19)
principalComponents = pca.fit_transform(x)

ax = plt.figure().gca()
ax.scatter(range(19), np.cumsum(pca.explained_variance_ratio_))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel("p component #")
plt.ylabel("explained variance ratio")

"""TODO:
* Given the above, let's try 4 or 5 PCA components. The first component will be most important and we should list the models from highest 1st component to lowest, and see if there is some sort of pattern. (Then for 2 and 3).

* Same story but with the dictionary learned below, skipping all zeros. (might even try 3 components or 2 to make it easier to explain.)
"""

from sklearn.decomposition import DictionaryLearning

# TODO: change params, fix random state
# alpha is sparsity
clf = DictionaryLearning(n_components=4, alpha=1.0, random_state=0)
x_transformed = clf.fit_transform(x)
x_transformed

for i, xcode in enumerate(x_transformed):
  print(model_names[i], xcode)

"""from sklearn.decomposition import NMF
n_componentss = list(range(2, 20))
errs = []
for n_components in n_componentss:
  clf = NMF(n_components=n_components, random_state = 0)
  principalComponents = clf.fit_transform(xorig)
  errs.append(clf.reconstruction_err_)

plt.plot(n_componentss, errs)

clf = NMF(n_components=5, random_state = 0)
principalComponents = clf.fit_transform(xorig)

clf.components_.shape

for i in range(5):
  print(i, sorted(list(zip(clf.components_[i,:].tolist(), task_names)), reverse=True))
"""

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

xp = principalDf['principal component 1']
yp = principalDf['principal component 2']

ax.scatter(xp, yp)

texts = []
for i, txt in enumerate(model_names):
    texts.append(ax.text(xp[i], yp[i], txt))
adjust_text(texts)

ax.grid()

"""### t-SNE

Model clustering

# Separating out the features
x = df_imputed_scaleavg_winsor.loc[:, :].values

from sklearn.manifold import TSNE

for perplexity in [5, 30, 50]:
  for learning_rate in [20.0, 62.0, 200.0, 500.0]:
    clf = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42, init='random')
    z = clf.fit_transform(x)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(f'tSNE perplexity={perplexity} lr={learning_rate}', fontsize = 20)

    xp = z[:,0]
    yp = z[:,1]
    ax.scatter(xp, yp)

    model_names = df_imputed_scaleavg_winsor.index.tolist()

    texts = []
    for i, txt in enumerate( model_names ):
        texts.append(ax.text(xp[i], yp[i], txt))
    adjust_text(texts)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#    ax.grid()
    fig.show()

This tSNE I guess looks reasonable.
"""

# Separating out the features
x = df_imputed_scaleavg_winsor.loc[:, :].values

from sklearn.manifold import TSNE

perplexity = 5
learning_rate = 500
clf = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42, init='random')
z = clf.fit_transform(x)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
#ax.set_title(f'tSNE perplexity={perplexity} lr={learning_rate}', fontsize = 20)

xp = z[:,0]
yp = z[:,1]
ax.scatter(xp, yp)

model_names = df_imputed_scaleavg_winsor.index.tolist()

texts = []
for i, txt in enumerate( model_names ):
    texts.append(ax.text(xp[i], yp[i], txt))
adjust_text(texts)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

#    ax.grid()
fig.show()

ax.figure.savefig("results/models-tsne.eps",bbox_inches='tight')
ax.figure.savefig("results/models-tsne.png",bbox_inches='tight')

"""Task clustering

from sklearn.manifold import TSNE

# Separating out the features
x = df_imputed_scaleavg_winsor.T.loc[:, :].values

for perplexity in [5, 30, 50]:
  for learning_rate in [20.0, 62.0, 200.0, 500.0]:
    clf = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42, init='random')
    z = clf.fit_transform(x)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_title(f'tSNE perplexity={perplexity} lr={learning_rate}', fontsize = 20)

    xp = z[:,0]
    yp = z[:,1]
    ax.scatter(xp, yp)

    task_names = df_imputed_scaleavg_winsor.T.index.tolist()

    texts = []
    for i, txt in enumerate(task_names):
        texts.append(ax.text(xp[i], yp[i], txt))
    adjust_text(texts)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#    ax.grid()

    fig.show()

We want ESC near FSD and gztan (see correlation table below) and maestro near nsynth.
"""

from sklearn.manifold import TSNE

# Separating out the features
x = df_imputed_scaleavg_winsor.T.loc[:, :].values

perplexity = 5
learning_rate = 500.0
clf = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42, init='random')
z = clf.fit_transform(x)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
#ax.set_title(f'tSNE perplexity={perplexity} lr={learning_rate}', fontsize = 20)

xp = z[:,0]
yp = z[:,1]
ax.scatter(xp, yp)

task_names = df_imputed_scaleavg_winsor.T.index.tolist()

texts = []
for i, txt in enumerate(task_names):
    texts.append(ax.text(xp[i], yp[i], txt))
adjust_text(texts)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

#    ax.grid()

fig.show()


ax.figure.savefig("results/tasks-tsne.eps",bbox_inches='tight')
ax.figure.savefig("results/tasks-tsne.png",bbox_inches='tight')

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

# Try different starting points to find best model to start from for TSP
from tqdm.auto import tqdm
from py2opt.routefinder import RouteFinder


def rotate(l, n):
    return l[-n:] + l[:-n]

import copy
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
from tqdm.auto import tqdm
from py2opt.routefinder import RouteFinder


def rotate(l, n):
    return l[-n:] + l[:-n]


import copy
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

"""## Archive results.zip"""

"""
!ls -l results/
!zip -r results-jekyll.zip results/

from google.colab import drive
drive.mount('/content/drive')
!mkdir  /content/drive/MyDrive/hear/
!cp results-jekyll.zip /content/drive/MyDrive/hear/


"""