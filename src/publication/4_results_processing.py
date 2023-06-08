github_local_folder_path = r'C:\Users\20190285\Documents\GitHub\hierarchicallearning/'

import pandas as pd
import numpy as np
import pickle
import os as os
import sys
sys.path.append(github_local_folder_path+'src')
# Utility functions
import hl.utils as utils
import hl.HierarchicalRegressor as regpkg


### Spatial or spatiotemporal dimensions

# Path Definition
path_in = github_local_folder_path + '/io/output/regressor/'
path_out = github_local_folder_path + '/io/output/results/'

dimension = 'spatial'
# dimension = 'spatiotemporal'
site = 'Fox'

prefix = 'BDG2_' + dimension + '_' + site
all_result_files = os.listdir(path_in)
result_files = [file for file in all_result_files if prefix in file and '_class' not in file and '_results' not in file
                and 'test' not in file and 'halfdaytree' not in file]
result_prefixes = [file.split('_regressor')[0] for file in result_files]


# Regressor class
filename = path_in + prefix
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


### Results processing
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

# Save information
df_res.to_csv(path_out+dimension+'_accuracy_perf.csv')
# coherency info
dic_df_res_matrix['coherency MS3E'].to_csv(path_out+dimension+'_coherency_perf.csv')





### Temporal dimension
dimension = 'temporal'
prefix = 'BDG2_' + dimension
all_result_files = os.listdir(path_in)
result_files = [file for file in all_result_files if prefix in file and '_class' not in file and '_results' not in file
                and 'test' not in file and 'halfdaytree' not in file]
result_prefixes = [file.split('_regressor')[0] for file in result_files]

# averaging temporal performances
dic_all_res_matrix = dict()
m_tot = 0
for prefix in result_prefixes:
    print(prefix)
    filename = path_in + prefix
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


key = 'global accuracy MS3E'
empty_df_line = pd.DataFrame(columns=dic_all_res_matrix[key].columns, index=[' '])
reformatted_result_matrix = pd.DataFrame(columns=dic_all_res_matrix[key].columns)

for col in [['None'], ['OLS', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV']]:
    df_block = dic_all_res_matrix[key].loc[col, :]
    reformatted_result_matrix = pd.concat([reformatted_result_matrix, df_block, empty_df_line], axis=0)
reformatted_result_matrix = reformatted_result_matrix.head(reformatted_result_matrix.shape[0] - 1)
df_res = reformatted_result_matrix.astype(np.float64)
dict_of_index_to_rename = {'OLS': 'id', 'STR': 'str', 'hVAR': 'hvar', 'sVAR': 'svar',
                             'COV': 'cov', 'kCOV': 'kcov'}
dict_of_columns_to_rename = {'OLS': 'hierarchical - id', 'STR': 'hierarchical - str', 'hVAR': 'hierarchical - hvar',
                             'sVAR': 'hierarchical - svar', 'COV': 'hierarchical - cov', 'kCOV': 'hierarchical - kcov'}
df_res.rename(index=dict_of_index_to_rename, columns=dict_of_columns_to_rename, inplace=True)

# Save information
df_res.to_csv(path_out+dimension+'_accuracy_perf.csv')
# coherency info
dic_all_res_matrix['coherency MS3E'].to_csv(path_out+dimension+'_coherency_perf.csv')





### Hierarchical time-series volatility extraction
filename = path_in + 'BDG2_spatial_Fox'
hreg = regpkg.H_Regressor.load(regpkg.H_Regressor, filename=filename, hierarchical_forecasting_method='OLS')

klvl = [key for key in hreg.tree.k_level_map.keys()]
klvl.sort(reverse=True)
resampling_rate = ['1H', '3H', '6H', '24H']

df_vol_res = pd.DataFrame(columns=resampling_rate)

for col in resampling_rate:
    for k in klvl:
        for id in hreg.tree.k_level_map[k]:
            std_dev = hreg.tree.df[id][['elec']].resample(col).mean().std().values[0]
            mean = hreg.tree.df[id][['elec']].resample(col).mean().mean().values[0]
            # coefficient of variation https://en.wikipedia.org/wiki/Coefficient_of_variation
            df_vol_res.loc[id, col] = std_dev / mean
df_vol_res = df_vol_res.apply(pd.to_numeric)

# Save information
df_vol_res.to_csv(path_out+'BDG2_volatility_perf.csv')
