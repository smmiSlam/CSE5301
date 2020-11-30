# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.hparams import *
from helpers.helpers import z_score, fill_missing

# Load dataset
data_df = pd.read_csv('../dataset/diabetes.csv')
raw_corrcoef = np.corrcoef(np.array(data_df).T)

# Fill in missing values
if FILL_DATA:
    fill_missing(data_df)

# Initialize scaled features
scaled_features = z_score(np.array(data_df))
# Visualize features with box-plots
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(12, 6))
ax[0, 0].boxplot(scaled_features)
ax[0, 0].set(
    xlabel='Features',
    ylabel='Feature Values',
    title='Five Point Descriptions of the Features'
)
ax[0, 0].set_xticks(np.arange(1, 10))
ax[0, 0].set_xticklabels(list(data_df.columns))
plt.grid(alpha=0.2)
plt.show()

# Visualize inter-correlations between features
fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(6, 6))
heatmap = sns.heatmap((np.corrcoef(scaled_features.T)-raw_corrcoef).round(2), annot=True, xticklabels=list(data_df.columns),
                      yticklabels=list(data_df.columns))
heatmap.set_yticklabels(
    heatmap.get_yticklabels(),
    rotation=20,
    horizontalalignment='right',
    fontweight='light'
)
heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=20,
    horizontalalignment='right',
    fontweight='light'
)
plt.tight_layout()
plt.show()
