github_local_folder_path = r'C:\Users\20190285\Documents\GitHub\hierarchicallearning/'

# Standard Libraries
import pandas as pd
import numpy as np
import time
from minepy import MINE
import scipy.cluster.hierarchy as sch
import glob
import sys
sys.path.append(github_local_folder_path+'src')
# Utility functions
import hl.utils as utils
import hl.TreeClass as treepkg
import hl.HierarchicalRegressor as regpkg
sys.setrecursionlimit(10000)

########################################################################################################################
print('Preprocessing \n')
########################################################################################################################
print(' \n Data PreProcessing phase \n')
print('Data Formatting \n')
# Path Definition
path_in = github_local_folder_path + '\io\input'
path_out = github_local_folder_path + '\io\output\regressor'

site_files = [f for f in glob.glob(path_in + '**_preproc.csv', recursive=True)]
sites = [site_file.split("\\")[8].split("_")[0] for site_file in site_files]
computed_sites = ['Bear', 'Bobcat', 'Bull', 'Cockatoo', 'Crow', 'Eagle', 'Fox', 'Gator', 'Hog', 'Lamb', 'Moose',
                  'Mouse', 'Panther', 'Peacock', 'Rat'] #Rat not done
# Gator not done coz only flat consumptions
sites = [site for site in sites if site not in computed_sites]

print('Data Integration \n')
#for site in sites:
site = 'Fox'
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
add_weather_info = True
for blg in buildings:

    if add_weather_info and len(weather_cols_still_there) > 0:
        df_tree[blg] = df_clean.loc[:, [blg]+weather_cols_still_there]
    else:
        df_tree[blg] = df_clean.loc[:, [blg]]

    df_tree[blg].rename(columns={blg: 'elec'}, inplace=True)
    add_weather_info = False

########################################################################################################################
print(' \n Mining: clustering')
########################################################################################################################

subset_of_buildings = buildings#[:50]
X = df_clean.loc[:, subset_of_buildings]
cluster_thresholds = {'Fox': 40000}
# Hierarchical clustering
Z = sch.linkage(X.T.values, method='ward', optimal_ordering=False)
leaves_labels = np.array(X.columns)
# Create S matrix from Z linkage matrix
S, y_ID = utils.create_Smatrix_from_linkagematrix(Z, leaves_labels)
threshold = 50000  #cluster_thresholds[site]

########################################################################################################################
print(' \n Hierarchy initialization')
########################################################################################################################
### Creating spatial tree
S, y_ID = utils.cluster_Smatrix_from_threshold(Z, S, y_ID, threshold=threshold)
tree_S = treepkg.Tree((S, y_ID), dimension='spatial')
# Creating data hierarchy
tree_S.create_spatial_hierarchy(df_tree, columns2aggr=['elec'])

### Creating temporal tree
tree_input_structure = {1: '1D', 2: '6H', 3: '3H', 4: '1H'}
#tree_input_structure = {1: '12H', 2: '4H', 3: '2H'}
tree_T = treepkg.Tree(tree_input_structure, dimension='temporal')
# # Creating data hierarchy
tree_T.create_temporal_hierarchy(df_tree[list(df_tree.keys())[1]], columns2aggr=['elec'])

### Creating Spatio-Temporal tree
tree_ST = treepkg.Multi_Tree(tree_S, tree_T)
tree_ST.create_ST_hierarchy(df_tree, columns2aggr=['elec'])

tree_H = tree_ST


extension = '_'

########################################################################################################################
print(' \n Feature Selection')
########################################################################################################################
# General approach:
# we only keep features with MIC scores above 0.25 per node ID
# then we extract automatically the top 3 PACF temporal shifts of the forecast target
target = 'elec'

### Maximal Information Coefficient
# https://medium.com/@rhondenewint93/on-maximal-information-coefficient-a-modern-approach-for-finding-associations-in-large-data-sets-ba8c36ebb96b
mine = MINE(alpha=0.6, c=15)

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
for n in tree_H.leaves_label:
    features[n] = list(tree_H.df[n].columns)
    features[n].remove(target[0])

########################################################################################################################
print(' \n Hierarchical Learning')
########################################################################################################################

# hierarchical forecasting methods:
hierarchical_forecasting_methods = ['multitask', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV']
hierarchical_forecasting_method = 'multitask'
# reconciliation methods: 'None', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV'
hierarchical_reconciliation_method = 'None'

hreg = regpkg.H_Regressor(tree_H, target_in=target, features_in=features,
                          forecasting_method=hierarchical_forecasting_method,
                          reconciliation_method=hierarchical_reconciliation_method,
                          alpha=0.75)
hreg.feature_engineering_wrapper(feature_toshift)

# Obtain training and testing sets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
nb_splits = 20
X_train, X_test, y_train, y_test, y_test_allflats = hreg.split_scaled(nb_splits=nb_splits, toscale=True,
                                                                      scaler=StandardScaler())



for hierarchical_forecasting_method in hierarchical_forecasting_methods:

    # Create the NN
    hreg.hierarchical_forecasting_method = hierarchical_forecasting_method
    hreg.create_ANN_network()

    ## Train the DANN
    for i in range(nb_splits):
        print(hierarchical_forecasting_method, i)
        # start the computing time tracking
        time_start = time.perf_counter()
        # For the first round, we initialise the G matrix as an OLS method - since we do not have prediction error
        # estimates yet
        hreg.scaler_attribute_update(divider=np.transpose(hreg.target_scalers['divider'][i]),
                                     shift=np.transpose(hreg.target_scalers['shift'][i]))
        if i == 0:
            hreg.coherency_constraint(y_diff=None)
            hreg.yhat_initialization(columns_in=target)
        # We train the DANN
        hreg.regressor.fit(X_train[i],
                           y_train[i],
                           epochs=500,
                           batch_size=24,
                           verbose=0,
                           shuffle=False)

        # We obtain the y_hat prediction over the test set
        y_hat = hreg.regressor.predict(X_test[i])
        # Save y_hat to a hierarchical df format
        y_hat_unscaled = hreg.inverse_scale_y(y_hat)
        y_diff = np.transpose(y_test_unscaled) - np.transpose(y_hat_unscaled)
        hreg.yhat = hreg.tree.reshape_flat2tree(flat_all_inputs=y_test_allflats[i], flat_data_in=y_hat_unscaled,
                                                tree_df_in=hreg.yhat)

        # finish the computing time tracking
        time_elapsed_forecast = (time.perf_counter() - time_start)

        # A-posteriori hard-constrained reconciliation
        for reconciliation_method in hreg.hierarchical_reconciliation_methods:
            time_start = time.perf_counter()
            y_tild = hreg.hard_ctr_reconciliation(y_diff=y_diff, y_hat=y_hat_unscaled,
                                                  method=reconciliation_method)
            time_elapsed_forecast_reconciliation = time_elapsed_forecast + (time.perf_counter() - time_start)

            # Saving the results
            hreg.save_performance_metrics(path_out + 'BDG2_' + hreg.tree.dimension + '_' + site + extension + '_results.txt',
                                          y_true=y_test_unscaled, y_hat=y_tild,
                                          comput_time=time_elapsed_forecast_reconciliation, iteration=i,
                                          reconciliation_method=reconciliation_method)
            print(hierarchical_forecasting_method, i, reconciliation_method)
        
        ## Update the coherency constraint in the loss function
        hreg.coherency_constraint(y_diff=y_diff)
        # We recompile the regressor to update the new loss function
        # see - https://stackoverflow.com/questions/60996892/how-to-replace-loss-function-during-training-tensorflow-keras
        loss_fct = hreg.coherency_loss_function_mse if hreg.hierarchical_forecasting_method != 'multitask' \
            else hreg.independent_loss_function_mse
        hreg.regressor.compile(loss=loss_fct,
                               optimizer='Adam',
                               metrics=['mae', 'mse'])


########################################################################################################################
print(' \n Class saving')
########################################################################################################################
filename = path_out + 'BDG2_' + hreg.tree.dimension + '_' + site + extension
# saving class and regressor objects /!\ careful this removes the regressor attribute form the current class
hreg.save(filename)