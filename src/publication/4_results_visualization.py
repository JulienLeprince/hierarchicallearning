github_local_folder_path = r'C:\Users\20190285\Documents\GitHub\hierarchicallearning/'

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os as os
import sys
sys.path.append(github_local_folder_path+'src')
# Utility functions
import hl.utils as utils
import hl.HierarchicalRegressor as regpkg

########################################################################################################################
print("Reading results \n")
########################################################################################################################
# Path Definition
path_in = github_local_folder_path + '\io\input'
path_out = github_local_folder_path + '\io\output\visuals'

dimension = 'temporal'
path_out = path_out + dimension + '/'
prefix = 'BDG2_' + dimension
all_result_files = os.listdir(path_in)
result_files = [file for file in all_result_files if prefix in file and '_class' not in file and '_results' not in file
                and 'test' not in file and 'halfdaytree' not in file]
result_prefixes = [file.split('_regressor')[0] for file in result_files]


for prefix in result_prefixes:
    #prefix = ''
    #prefix = 'BDG2_' + dimension + '_Fox'

    # Regressor class
    filename = path_in + prefix #+ '_regressor'
    hreg = regpkg.H_Regressor.load(regpkg.H_Regressor, filename=filename, hierarchical_forecasting_method='OLS')

    # Results metrics
    filename = path_in + prefix + '_results.txt'
    fileobject = open(filename, 'rb', )
    dic_df_res = pickle.load(fileobject)
    fileobject.close()

    global_perf_metrics = ['global accuracy MSE', 'global accuracy MS3E', 'coherency MSE', 'coherency MS3E',
                           'computation', 'global_successful_forecast', 'global_y_hat_std', 'global_y_hat_mean']
    pernode_perf_metrics = ['successful_forecast', 'y_hat_std', 'y_hat_mean', 'accuracy MSE', 'accuracy MS3E']

    perf_metrics_saved = list(dic_df_res.keys())
    perf_metrics_saved.remove('m')
    for node_metric in pernode_perf_metrics:
        perf_metrics_saved.remove(node_metric)

    ########################################################################################################################
    print("Results processing \n")
    ########################################################################################################################

    ## Create unique metric matrix from dic_df_res
    results_f_cols = [str(i) for i in hreg.hierarchical_forecasting_methods]
    results_r_cols = [str(j) for j in hreg.hierarchical_reconciliation_methods]
    dic_df_res_matrix = dict()
    for key in perf_metrics_saved:
        dic_df_res_matrix[key] = pd.DataFrame(0, index=results_r_cols, columns=results_f_cols)

    # Attribute metric value from dic_df_res object
    for key in perf_metrics_saved:
        for column in dic_df_res[key].columns:
            forecasting_method = column.split('_')[0]
            reconciliation_method = column.split('_')[1]
            col = str(forecasting_method)
            idx = str(reconciliation_method)
            dic_df_res_matrix[key].loc[idx, col] = dic_df_res[key][column].mean()
        dic_df_res_matrix[key].drop('BU', axis=1, inplace=True)
        dic_df_res_matrix[key].drop('BU', axis=0, inplace=True)

    ########################################################################################################################
    print("Visualization \n")
    ########################################################################################################################
    hfont = {'fontname': 'Times New Roman'}

    # global_perf_metrics = ['global accuracy MSE', 'global accuracy MS3E', 'coherency MSE', 'coherency MS3E',
    #                        'computation', 'global_successful_forecast', 'global_y_hat_std', 'global_y_hat_mean']
    # pernode_perf_metrics = ['successful_forecast', 'y_hat_std', 'y_hat_mean', 'accuracy MSE', 'accuracy MS3E']

    key = 'global accuracy MS3E'  # accuracy MSE, accuracy MS3E, coherency MSE, coherency MS3E, computation

    ### Result plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im, cbar = utils.heatmap(dic_df_res_matrix[key].values,
                             dic_df_res_matrix[key].index,
                             dic_df_res_matrix[key].columns, ax=ax,
                             cmap='summer', cbarlabel='MS3E [kWh]')
    texts = utils.annotate_heatmap(im, **hfont)
    ax.set_ylabel('Reconciliation', fontsize=14, **hfont)
    plt.gca().xaxis.set_label_position('top')
    ax.set_xlabel('Forecasting', fontsize=14, **hfont)
    ax.set_title('n = ' + str(hreg.tree.n) + ', m = ' + str(hreg.tree.m), fontsize=18, **hfont)

    fig.tight_layout()
    #plt.show()

    plt.savefig(path_out + prefix + '_' + key + 'results.png')
    plt.close()






## PUBLICATION PLOT

import numpy as np
hfont = {'fontname': 'Times New Roman'}
key = 'global accuracy MS3E'
empty_df_line = pd.DataFrame(columns=dic_df_res_matrix[key].columns, index=[' '])
reformatted_result_matrix = pd.DataFrame(columns=dic_df_res_matrix[key].columns)

for col in [['None'], ['OLS', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV']]:
    df_block = dic_df_res_matrix[key].loc[col, :]
    reformatted_result_matrix = pd.concat([reformatted_result_matrix, df_block, empty_df_line], axis=0)
reformatted_result_matrix = reformatted_result_matrix.head(reformatted_result_matrix.shape[0] - 1)
df_res = reformatted_result_matrix.astype(np.float64)
dict_of_index_to_rename = {'OLS': 'id', 'STR': 'str', 'hVAR': 'hvar', 'sVAR': 'svar',
                             'COV': 'cov', 'kCOV': 'kcov'}
dict_of_columns_to_rename = {'OLS': 'hierarchical - id', 'STR': 'hierarchical - str', 'hVAR': 'hierarchical - hvar',
                             'sVAR': 'hierarchical - svar', 'COV': 'hierarchical - cov', 'kCOV': 'hierarchical - kcov'}
df_res.rename(index=dict_of_index_to_rename, columns=dict_of_columns_to_rename, inplace=True)

### Subplots
from matplotlib import gridspec
import math
import itertools
def rof(x):
  '''round up to multiple of 5'''
  if x%5==4:
      x+=1
  elif x%5==3:
      x+=2
  return x

from copy import copy

# Subplots
fig = plt.figure(constrained_layout=True, figsize=(8, 7))
my_cmap = copy(plt.cm.summer)
my_cmap.set_over("white")
threshold = 8.7e4

spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig,
                         height_ratios=[1.2, 1, 5],
                         #width_ratios=[1],
                         left=0, right=1, hspace=0,
                         )

axs1 = fig.add_subplot(spec[0, 0])
plt.box(on=None)
axs2 = fig.add_subplot(spec[1, 0], sharex=axs1)
plt.tick_params(labelbottom=False, bottom=False)
plt.yticks(**hfont)
plt.xticks(**hfont)
axs3 = fig.add_subplot(spec[2, 0], sharex=axs1)

## Count plot
counts = dic_df_res_matrix['coherency MS3E'].loc[['None'], :]
index = dic_df_res_matrix['coherency MS3E'].columns

axs1.bar(index, counts.values[0], color='k', zorder=3)

# Legend
axs1.grid(axis='y', linestyle='--', zorder=0)
# Ticks
tick_list = np.arange(0, int(round(max(counts.values[0]), -3)), int(round(max(counts.values[0])/5, -3)))
axs1.set_yticks(tick_list)
nan_list = ([0, np.nan]*math.floor(len(tick_list)/2))
tick_label = [sum(x) for x in itertools.zip_longest(*[tick_list, nan_list], fillvalue=0)]
tick_label = [None if np.isnan(x) else x for x in tick_label]
axs1.set_yticklabels(tick_label, fontsize=12)
axs1.xaxis.set_visible(False)
axs1.set_ylabel('$\mathcal{L}$ᶜ [kWh]', fontsize=11, **hfont)
axs1.set_ylim(0, max(counts.values[0])+80)

# source: https://www.geeksforgeeks.org/how-to-annotate-bars-in-barplot-with-matplotlib-in-python/
for bar in axs1.patches:
    height = bar.get_height()
    axs1.annotate(format(bar.get_height(), '.4g'),
                  (bar.get_x() + bar.get_width() / 2,
                   height), ha='center', va='center',
                   size=11, xytext=(0, 8),
                   textcoords='offset points', **hfont)

# Heatmap plot 1
heatmap_color = 'summer'  # summer, viridis
im1 = axs2.imshow(df_res.loc[['None'], :].values,
                  vmax=threshold, #df_res.max().max(),
                  vmin=df_res.min().min(),
                  aspect='auto',
                  cmap=my_cmap)#heatmap_color)
axs2.set_yticks(np.arange(df_res.loc[['None'], :].values.shape[0]), labels=df_res.loc[['None'], :].index)
texts = utils.annotate_heatmap(im1, threshold=threshold/2, **hfont)
axs2.spines[:].set_visible(False)

axs2.set_xticks(np.arange(df_res.loc[['None'], :].values.shape[1] + 1) - .5, minor=True)
axs2.set_yticks(np.arange(df_res.loc[['None'], :].values.shape[0] + 1) - .5, minor=True)
axs2.grid(which="minor", color="w", linestyle='-', linewidth=3)
axs2.tick_params(which="minor", bottom=False, left=False)

# Heatmap plot 2
im2, cbar = utils.heatmap(df_res.loc[['id', 'str', 'hvar', 'svar', 'cov', 'kcov'], :].values,
                          df_res.loc[['id', 'str', 'hvar', 'svar', 'cov', 'kcov'], :].index,
                          df_res.columns, ax=axs3,
                          vmax=threshold, #df_res.max().max(),
                          vmin=df_res.min().min(),
                          aspect='auto',
                          cmap=my_cmap,#heatmap_color,
                          cbarlabel='$\mathcal{L}$ʰ MS3E [kWh]')
texts = utils.annotate_heatmap(im2, threshold=threshold/2, **hfont)

axs3.set_ylabel('Reconciliation method', fontsize=14, labelpad=9, **hfont)
plt.gca().xaxis.set_label_position('bottom')
axs3.set_xlabel('Forecasting method', fontsize=14, labelpad=9, **hfont)

plt.savefig(path_out + prefix + '_manuscript_plot.pdf')
plt.close()






# Plot standard
fig, ax = plt.subplots(figsize=(10, 8))
im, cbar = utils.heatmap(df_res.values,
                         df_res.index,
                         df_res.columns, ax=ax,
                         cmap='summer', cbarlabel='MS3E [kWh]')
texts = utils.annotate_heatmap(im, **hfont)
yticks = ax.yaxis.get_major_ticks()
yticks[0].set_visible(False)
yticks[1].set_visible(False)

ax.set_ylabel('Reconciliation                           ', fontsize=14, labelpad=9, **hfont)
plt.gca().xaxis.set_label_position('bottom')
ax.set_xlabel('Forecasting', fontsize=14, **hfont)
ax.set_title('n = ' + str(hreg.tree.n) + ', m = ' + str(hreg.tree.m), fontsize=18, **hfont)

fig.tight_layout()










# dic_df_res['global accuracy MS3E']['multitask_None']
# dic_df_res['global accuracy MS3E']['sVAR_None']
# result_prefixes.remove('BDG2_temporal_Fox_education_Andre')

########################################################################################################################
print("All performances averaged \n")
########################################################################################################################
dic_all_res_matrix = dict()
m_tot = 0
#for site in computed_sites:
for prefix in result_prefixes:
    print(prefix)
    filename = path_in + prefix  # + '_regressor'

    # prefix = 'BDG2_' + 'spatial' + '_' + site
    # # Regressor class
    # filename = path_in + prefix + '_regressor.txt'
    hreg = regpkg.H_Regressor.load(regpkg.H_Regressor, filename=filename, hierarchical_forecasting_method='OLS')

    # Results metrics
    filename = path_in + prefix + '_results.txt'
    fileobject = open(filename, 'rb', )
    dic_df_res = pickle.load(fileobject)
    fileobject.close()

    global_perf_metrics = ['global accuracy MSE', 'global accuracy MS3E', 'coherency MSE', 'coherency MS3E',
                           'computation', 'global_successful_forecast', 'global_y_hat_std', 'global_y_hat_mean']
    pernode_perf_metrics = ['successful_forecast', 'y_hat_std', 'y_hat_mean', 'accuracy MSE', 'accuracy MS3E']

    perf_metrics_saved = list(dic_df_res.keys())
    perf_metrics_saved.remove('m')
    for node_metric in pernode_perf_metrics:
        perf_metrics_saved.remove(node_metric)

    results_f_cols = [str(i) for i in hreg.hierarchical_forecasting_methods]
    results_r_cols = [str(j) for j in hreg.hierarchical_reconciliation_methods]
    dic_df_res_matrix = dict()
    for key in perf_metrics_saved:
        dic_df_res_matrix[key] = pd.DataFrame(0, index=results_r_cols, columns=results_f_cols)

    # Attribute metric value from dic_df_res object
    for key in perf_metrics_saved:
        for column in dic_df_res[key].columns:
            forecasting_method = column.split('_')[0]
            reconciliation_method = column.split('_')[1]
            col = str(forecasting_method)
            idx = str(reconciliation_method)
            dic_df_res_matrix[key].loc[idx, col] = dic_df_res[key][column].mean()
        dic_df_res_matrix[key].drop('BU', axis=1, inplace=True)
        dic_df_res_matrix[key].drop('BU', axis=0, inplace=True)

        if key not in dic_all_res_matrix.keys():
            dic_all_res_matrix[key] = dic_df_res_matrix[key]*hreg.tree.m if key == 'm' \
                else dic_df_res_matrix[key]*hreg.tree.n
        else:
            dic_all_res_matrix[key] = dic_all_res_matrix[key] + dic_df_res_matrix[key]*hreg.tree.m
    m_tot = m_tot + hreg.tree.m

for key in dic_all_res_matrix:
    dic_all_res_matrix[key] = dic_all_res_matrix[key]/m_tot




from matplotlib import gridspec
import math
import itertools
import numpy as np
hfont = {'fontname': 'Times New Roman'}
key = 'global accuracy MS3E'
dict_of_index_to_rename = {'OLS': 'id', 'STR': 'str', 'hVAR': 'hvar', 'sVAR': 'svar',
                             'COV': 'cov', 'kCOV': 'kcov'}
dict_of_columns_to_rename = {'OLS': 'hierarchical - id', 'STR': 'hierarchical - str', 'hVAR': 'hierarchical - hvar',
                             'sVAR': 'hierarchical - svar', 'COV': 'hierarchical - cov', 'kCOV': 'hierarchical - kcov'}
dic_all_res_matrix[key].rename(index=dict_of_index_to_rename, columns=dict_of_columns_to_rename, inplace=True)




## Publication plot
fig = plt.figure(constrained_layout=True, figsize=(8, 7))
spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig,
                         height_ratios=[1.2, 1, 5],
                         left=0, right=1, hspace=0,
                         )

axs1 = fig.add_subplot(spec[0,0])
plt.box(on=None)
axs2 = fig.add_subplot(spec[1,0], sharex=axs1)
plt.tick_params(labelbottom=False, bottom=False)
plt.yticks(**hfont)
plt.xticks(**hfont)
axs3 = fig.add_subplot(spec[2,0], sharex=axs1)

## Count plot
counts = dic_all_res_matrix['coherency MS3E'].loc[['None'], :]
index = dic_all_res_matrix['coherency MS3E'].columns
df_res = dic_all_res_matrix[key]

axs1.bar(index, counts.values[0], color='k', zorder=3)

# Legend
axs1.grid(axis='y', linestyle='--', zorder=0)
# Ticks
tick_list = np.arange(0, int(round(max(counts.values[0]), -3)), int(round(max(counts.values[0])/5, -3)))
axs1.set_yticks(tick_list)
nan_list = ([0, np.nan]*math.floor(len(tick_list)/2))
tick_label = [sum(x) for x in itertools.zip_longest(*[tick_list, nan_list], fillvalue=0)]
tick_label = [None if np.isnan(x) else x for x in tick_label]
axs1.set_yticklabels(tick_label, fontsize=12)
axs1.xaxis.set_visible(False)
axs1.set_ylabel('$\mathcal{L}$ᶜ [kWh]', fontsize=11, **hfont)
axs1.set_ylim(0, max(counts.values[0])+80)

# source: https://www.geeksforgeeks.org/how-to-annotate-bars-in-barplot-with-matplotlib-in-python/
for bar in axs1.patches:
    # if bar.get_height() == max([b.get_height() for b in axs1.patches]):
    #     height = bar.get_height() - 40
    # else:
    #     height = bar.get_height()
    height = bar.get_height()
    axs1.annotate(format(bar.get_height(), '.4g'),
                  (bar.get_x() + bar.get_width() / 2,
                   height), ha='center', va='center',
                   size=11, xytext=(0, 8),
                   textcoords='offset points', **hfont)

# ['None'], ['OLS', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV']
# Heatmap plot 1
im1 = axs2.imshow(df_res.loc[['None'], :].values,
                  vmax=df_res.max().max(), vmin=df_res.min().min(),
                  aspect='auto',
                  cmap='summer')
axs2.set_yticks(np.arange(df_res.loc[['None'], :].values.shape[0]),
                labels=df_res.loc[['None'], :].index)
texts = utils.annotate_heatmap(im1, **hfont)
axs2.spines[:].set_visible(False)

axs2.set_xticks(np.arange(df_res.loc[['None'], :].values.shape[1] + 1) - .5, minor=True)
axs2.set_yticks(np.arange(df_res.loc[['None'], :].values.shape[0] + 1) - .5, minor=True)
axs2.grid(which="minor", color="w", linestyle='-', linewidth=3)
axs2.tick_params(which="minor", bottom=False, left=False)

# Heatmap plot 2
im2, cbar = utils.heatmap(df_res.loc[['id', 'str', 'hvar', 'svar', 'cov', 'kcov'], :].values,
                          df_res.loc[['id', 'str', 'hvar', 'svar', 'cov', 'kcov'], :].index,
                          df_res.columns, ax=axs3,
                          vmax=df_res.max().max(), vmin=df_res.min().min(),
                          aspect='auto',
                          cmap='summer', cbarlabel='$\mathcal{L}$ʰ MS3E [kWh]')
texts = utils.annotate_heatmap(im2, **hfont)

axs3.set_ylabel('Reconciliation method', fontsize=14, labelpad=9, **hfont)
plt.gca().xaxis.set_label_position('bottom')
axs3.set_xlabel('Forecasting method', fontsize=14, labelpad=9, **hfont)


path_out = r'C:\Users\20190285\surfdrive\05_Data\054_inout\0546_HPL\visuals/temporal_24H/'
plt.savefig(path_out + 'BDG2_temporal_all_manuscript_plot.pdf')
plt.close()
