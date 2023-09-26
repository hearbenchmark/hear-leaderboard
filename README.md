hear-leaderboard
================

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

```
pip install py2opt
```
