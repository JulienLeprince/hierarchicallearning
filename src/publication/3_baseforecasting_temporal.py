github_local_folder_path = r'C:\Users\20190285\Documents\GitHub\hierarchicallearning/'

# Standard Libraries
import pandas as pd
import numpy as np
import time
from minepy import MINE
import scipy.cluster.hierarchy as sch
import glob
import os
import sys
sys.path.append(github_local_folder_path+'src')
# Utility functions
import hl.utils as utils
import hl.TreeClass as treepkg
import hl.HierarchicalRegressor as regpkg


########################################################################################################################
print("Preprocessing \n")
########################################################################################################################
print(" \n Data PreProcessing phase \n")
print("Data Formatting \n")
# Path Definition
path_in = github_local_folder_path + '\io\input'
path_out = github_local_folder_path + '\io\output\regressor'

site_files = [f for f in glob.glob(path_in + '**_preproc.csv', recursive=True)]
sites = [site_file.split("\\")[8].split("_")[0] for site_file in site_files]
computed_sites = ['Gator', 'Rat', 'Wolf', 'Bear']
# Gator not done coz only flat consumptions, and Rat is too big
sites = [site for site in sites if site not in computed_sites]

print('Data Integration \n')
site = 'Fox'
#for site in sites:
site_file = [site_file for site_file in site_files if site_file.split('\\')[8].split('_')[0] == site][0]
site = site_file.split('\\')[8].split('_')[0]
# Reading excel files (duplicates identification done in read function)
df = pd.read_csv(site_file, index_col=[0])

# Timestamps to datetime
df.index = pd.to_datetime(df.index)
delta_t = pd.to_timedelta(df.index[1]-df.index[0])

weather_cols = ['airTemperature', 'windSpeed', 'precipDepth1HR']
# Building IDs
buildings = df.columns.tolist()
for col in weather_cols:
    buildings.remove(col)

# Drop columns with all NaNs
df.dropna(axis=1, how='all', inplace=True)
weather_cols_still_there = [col for col in df.columns.tolist() if col in weather_cols]

# All timestamp intervals
timestamp = pd.date_range(start=df.index[0], end=df.index[-1], freq='H')
df_newindex = pd.DataFrame(0, index=timestamp, columns=['todrop'])
df = pd.concat([df, df_newindex], axis=1)
df.drop('todrop', axis=1, inplace=True)
df.interpolate(method='time', inplace=True)
df.index.set_names('timestamp', inplace=True)

t_itv = utils.inner_intersect(df.reset_index(), col_ref=df.columns[0], col_loop=df.columns[1::])
# Selecting the longest of them
idx_max = t_itv['Consecutive_Measurements'].astype(float).idxmax()
# Keep only largest consecutive time interval without NaNs
df_clean = df.loc[t_itv['BeginDate'][idx_max]: t_itv['EndDate'][idx_max]]

# Interpolate and drop columns with missing vals
df_clean.interpolate(method='time', inplace=True)
df_clean.dropna(axis=1, inplace=True)

# Filter out flat consumptions
variations = df_clean.diff()
pourcentage_of_null_variations = variations[variations == 0].count() / len(variations)
flat_columns = [col for col in variations.columns if pourcentage_of_null_variations[col] > 0.30]
df_clean.drop(flat_columns, axis=1, inplace=True)

buildings = df_clean.columns.tolist()

########################################################################################################################
print('Formating \n')
########################################################################################################################

df_tree = dict()
for blg in buildings:

    if len(weather_cols_still_there) > 0:
        df_tree[blg] = df_clean.loc[:, [blg]+weather_cols_still_there]
    else:
        df_tree[blg] = df_clean.loc[:, [blg]]

    df_tree[blg].rename(columns={blg: 'elec'}, inplace=True)

########################################################################################################################
print(' \n Mining: clustering')
########################################################################################################################

X = df_clean.loc[:, buildings]
cluster_thresholds = {'Fox': 40000}

### Hierarchical clustering
Z = sch.linkage(X.T.values, method='ward', optimal_ordering=False)
leaves_labels = np.array(X.columns)
# Create S matrix from Z linkage matrix
S, y_ID = utils.create_Smatrix_from_linkagematrix(Z, leaves_labels)
threshold = cluster_thresholds[site]

########################################################################################################################
print(' \n Hierarchy initialization')
########################################################################################################################

### Creating temporal tree
tree_input_structure = {1: '1D', 2: '6H', 3: '3H', 4: '1H'}
#tree_input_structure = {1:'12H', 2:'6H'}
tree_T = treepkg.Tree(tree_input_structure, dimension='temporal')


# Identifying flies to loop over
all_result_files = os.listdir(path_out)
result_files = [file for file in all_result_files if '_temporal' in file and 'regressor' in file]
sites_computed = [file.split('_')[2] + '_' + file.split('_')[3] + '_' + file.split('_')[4] for file in result_files]
sites_2_compute = [site for site in list(df_tree.keys()) if site not in sites_computed]


for i, site in enumerate(sites_2_compute):
    print(i, len(sites_2_compute)-1, site)
    # # Creating data hierarchy
    tree_T.create_temporal_hierarchy(df_tree[site], columns2aggr=['elec'])
    tree_H = tree_T

    ########################################################################################################################
    print(" \n Feature Selection")
    ########################################################################################################################
    target = 'elec'
    # General approach:
    # we only keep features with MIC scores above 0.25 per node ID
    # then we extract automatically the top 3 PACF temporal shifts of the forecast target

    ### Maximal Information Coefficient
    # https://medium.com/@rhondenewint93/on-maximal-information-coefficient-a-modern-approach-for-finding-associations-in-large-data-sets-ba8c36ebb96b
    mine = MINE(alpha=0.6, c=15)
    target = 'elec'

    # We drop feature displaying a MIC score below |+- 25|
    for node in tree_H.df:
        for col in tree_H.df[node]:
            mine.compute_score(tree_H.df[node][target], tree_H.df[node][col])
            if np.abs(mine.mic()) < 0.25:
                tree_H.df[node].drop(col, axis=1)

    ### PACF
    # https://stackoverflow.com/questions/62735128/is-there-a-way-to-extract-the-points-from-a-p-acf-graph-in-python
    import statsmodels.api as sm
    feature_toshift = {}

    if tree_H.dimension == 'spatial':
        for node in tree_H.df:
            pac_largest_timedeltas = utils.get_pac_largest_timedeltas(tree_H.df[node][target], n=3, threshold=0.25)
            feature_toshift[node] = {target: pac_largest_timedeltas}

    elif tree_H.dimension == 'temporal':
        for k in tree_H.k_level_map:
            pac_largest_timedeltas = utils.get_pac_largest_timedeltas(tree_H.df_klvl(k)[target], n=3, threshold=0.25)
            feature_toshift[k] = {target: pac_largest_timedeltas}

    elif tree_H.dimension == 'spatiotemporal':
        for node in tree_H.spatial.y_ID:
            for k in tree_H.temporal.k_level_map:
                pac_largest_timedeltas = utils.get_pac_largest_timedeltas(tree_H.df_klvl(k, node)[target], n=3,
                                                                          threshold=0.25)
                feature_toshift[(node, k)] = {target: pac_largest_timedeltas}


    for n in feature_toshift:
        print(n, feature_toshift[n])

    ### Defining target, features and feature shifts
    target = [target]
    features = dict()
    for n in tree_H.y_ID:
        features[n] = list(tree_H.df[n].columns)
        features[n].remove(target[0])

    ########################################################################################################################
    print(" \n Hierarchical Learning")
    ########################################################################################################################

    # reconciliation methods: 'None', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV'
    hierarchical_reconciliation_method = 'None'

    ### Creating base Predictive Learning object wraper
    breg = regpkg.Base_Regressor(tree_H, target_in=target, features_in=features,
                                 forecasting_method='base',
                                 reconciliation_method=hierarchical_reconciliation_method)
    breg.coherency_constraint(method=breg.reconciliation_method)

    # Create the NN
    breg.feature_engineering_wrapper(feature_toshift, all_temporal_nodes=True, all_spatial_nodes=True,
                                     auto_adapt_shifts=False)
    breg.regressor = dict()
    for n in breg.tree.y_ID:
        breg.regressor[n] = breg.create_base_ANN_network(n)


    # Obtain training and testing sets
    # define only_leaves_X as False to obtain full tree in X input sets rather than only leaves
    nb_splits = 20
    X_train, X_test, y_train, y_test, y_test_allflats = breg.split_scaled(nb_splits=nb_splits,
                                                                          toscale=True,
                                                                          only_leaves_X=False)

    ## Train the DANN
    for i in range(nb_splits):

        y_hat, y_hatrain = [], []
        if i == 0:
            breg.yhat_initialization(columns_in=target)
        index_start = 0
        breg.scaler_attribute_update(divider=np.transpose(breg.target_scalers['divider'][i]),
                                     shift=np.transpose(breg.target_scalers['shift'][i]))

        for ni, n in enumerate(breg.tree.y_ID):
            print('loop ', i, '/', nb_splits, '- node: ', ni, ' up till ', len(breg.tree.y_ID), n)
            # start the computing time tracking
            time_start = time.perf_counter()
            # We train the DANN
            index_end = index_start + len(features[n])
            breg.regressor[n].fit(np.transpose(np.transpose(X_train[i])[index_start: index_end]),
                                  np.transpose(y_train[i])[ni],
                                  epochs=500,
                                  batch_size=24,
                                  verbose=0,
                                  shuffle=False)

            # We obtain the y_hat prediction over the test set
            y_hati = breg.regressor[n].predict(np.transpose(np.transpose(X_test[i])[index_start: index_end]))
            y_hatraini = breg.regressor[n].predict(np.transpose(np.transpose(X_train[i])[index_start: index_end]))
            # finish the computing time tracking
            time_elapsed_forecast = (time.perf_counter() - time_start)
            y_hat.append(utils.flatten(list(y_hati.T)))
            y_hatrain.append(utils.flatten(list(y_hatraini.T)))

            index_start = index_end

        y_hat = utils.flatten(y_hat)
        y_hatrain = utils.flatten(y_hatrain)
        y_hat_unscaled = breg.inverse_scale_y(np.transpose(y_hat))

        # Save y_hat to a hierarchical df format
        breg.yhat = breg.tree.reshape_flat2tree(flat_all_inputs=y_test_allflats[i],
                                                flat_data_in=y_hat_unscaled,
                                                tree_df_in=breg.yhat)

        # Inverse scaling y vectors for reconciliation
        y_test_unscaled = breg.inverse_scale_y(y_test[i])
        y_hatrain_unscaled = breg.inverse_scale_y(np.transpose(y_hatrain))
        y_train_unscaled = breg.inverse_scale_y(y_train[i])
        y_diff = np.transpose(y_train_unscaled) - np.transpose(y_hatrain_unscaled)

        # A-posteriori hard-constrained reconciliation
        for reconciliation_method in breg.hierarchical_reconciliation_methods:
            time_start = time.perf_counter()
            y_tild = breg.hard_ctr_reconciliation(y_diff=y_diff, y_hat=y_hat_unscaled,
                                                  method=reconciliation_method)
            time_elapsed_forecast_reconciliation = time_elapsed_forecast + (time.perf_counter() - time_start)

            # Saving the results
            breg.save_performance_metrics(path_out + 'BDG2_' + breg.tree.dimension + '_' + site + '_results.txt',
                                          y_true=np.array(y_test_unscaled),
                                          y_hat=np.array(y_tild),
                                          comput_time=time_elapsed_forecast_reconciliation, iteration=i,
                                          reconciliation_method=reconciliation_method)
            print(i, reconciliation_method)