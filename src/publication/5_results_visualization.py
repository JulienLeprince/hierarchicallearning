# Note: This file does not require Tensorflow to be installed in the python interpreter, only seaborn and matplotlib
import pandas as pd
import numpy as np
import sys

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


# Path Definition
github_local_folder_path = r'C:\Users\20190285\Documents\GitHub\hierarchicallearning/'
path_in = github_local_folder_path + '/io/output/results/'
path_out = github_local_folder_path + '/io/output/visuals/'

dimension = 'spatial'
prefix = 'BDG2_' + dimension

# Read
df_accuracy = pd.read_csv(path_in+dimension+'_accuracy_perf.csv', index_col=0)
df_coherency = pd.read_csv(path_in+dimension+'_coherency_perf.csv', index_col=0)

def rof(x):
  '''round up to multiple of 5'''
  if x%5==4:
      x+=1
  elif x%5==3:
      x+=2
  return x


# Forecasting performance plot
hfont = {'fontname': 'Times New Roman'}
sns.set(rc={"figure.figsize": (7, 5)})
sns.set_theme(style="whitegrid", font='serif', font_scale=.8)

# https://stackoverflow.com/questions/65037113/seaborn-heatmaps-in-subplots-align-x-axis
fig, ax = plt.subplots(2, 2, sharex='col',
                        gridspec_kw=dict(width_ratios=[100, 5], height_ratios=[2, 8]))
plt.subplots_adjust(wspace=0.1, hspace=0.02)
ax[0,1].remove()

#Bar plot
bar_plot_xaxis_adjustment = 0.5
ax[0,0].bar(np.arange(len(df_accuracy.columns))+bar_plot_xaxis_adjustment, df_coherency.loc['None'], color='0.8')
ax[0,0].set_ylabel('$\mathcal{L}$ᶜ [kWh]')
for x, temp in enumerate(df_coherency.loc['None']):
    ax[0,0].annotate('{:.3g}'.format(temp), xy=(x+bar_plot_xaxis_adjustment, temp), xytext=(0, 2),
                     xycoords='data', textcoords="offset points",  ha='center', va='bottom', fontsize=8)
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html
#ax[0,0].set_xticks(np.arange(len(df_accuracy.columns))+0.5)
ax[0,0].spines['top'].set_visible(False)
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['bottom'].set_visible(False)
ax[0,0].spines['left'].set_visible(False)
ax[0,0].xaxis.set_visible(False)
ax[0,0].grid(axis='y', linestyle='--', zorder=0)

# Heatmap
sns.heatmap(df_accuracy, cmap="BuPu", linewidths=.6, square=False, norm=LogNorm(),
            fmt='.3g', annot_kws={"size": 8},
            annot=True, ax=ax[1,0], cbar_ax=ax[1,1],
            cbar_kws={'label': 'MS3E [kWh]'})
# https://stackoverflow.com/questions/62696868/highlighting-maximum-value-in-a-column-on-a-seaborn-heatmap
xpos = df_accuracy.index.get_loc(df_accuracy[min(df_accuracy)].idxmin())
ypos = df_accuracy.columns.get_loc(min(df_accuracy))
ax[1,0].add_patch(Rectangle((ypos,xpos),1,1, fill=False, edgecolor='red', lw=.5))
ax[1,0].xaxis.tick_bottom()  # x axis on top
ax[1,0].xaxis.set_label_position('bottom')
ax[1,0].set_xticklabels(df_accuracy.columns, rotation=90)
ax[1,0].set_yticklabels(df_accuracy.index, rotation=0)
ax[1,0].set_xlabel('Forecasting method', fontsize=14, labelpad=9, **hfont)
ax[1,0].set_ylabel('Reconciliation method', fontsize=14, labelpad=9, **hfont)

plt.gca().xaxis.set_label_position('bottom')
plt.tight_layout()
plt.savefig(path_out + prefix + '_manuscript_plot.pdf')
plt.close()



# Volatility plot
df_vol_res = pd.read_csv(path_in+ 'BDG2_volatility_perf.csv', index_col=0)
df_vol_res.sort_values('1H', ascending=True, inplace=True)

hfont = {'fontname': 'Times New Roman'}
sns.set(rc={"figure.figsize": (4, 12)})
sns.set_theme(style="whitegrid", font='serif', font_scale=.8)
# https://seaborn.pydata.org/generated/seaborn.axes_style.html

ax = sns.heatmap(df_vol_res, cmap="BuPu", linewidths=.6, norm=LogNorm(),
                 #annot=True,
                 cbar_kws={'label': 'coefficient of variation $\sigma / \mu$ [-]'})
ax.xaxis.tick_top()  # x axis on top
ax.xaxis.set_label_position('top')
ax.set_ylabel('tree node identification', fontsize=12, labelpad=9, **hfont)
ax.set_xlabel('resampling rate', fontsize=12, labelpad=9, **hfont)

plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(path_out + 'BDG2_variability_manuscript_plot.pdf')
plt.close()
