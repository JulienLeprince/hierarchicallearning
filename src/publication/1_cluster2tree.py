github_local_folder_path = r'C:\Users\20190285\Documents\GitHub\hierarchicallearning/'

# Standard Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import glob

import sys
sys.path.append(github_local_folder_path+'src')
# Utility functions
import hl.utils as utils
import hl.TreeClass as treepkg


########################################################################################################################
print("Preprocessing \n")
########################################################################################################################
print(" \n Data PreProcessing phase \n")
print("Data Formatting \n")
# Path Definition
path_in = github_local_folder_path + '\io\input'
site_files = [f for f in glob.glob(path_in + "**_preproc.csv", recursive=True)]


print("Data Integration \n")  #######################################################################
site_file = site_files[0]

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

t_itv = utils.inner_intersect(df.reset_index(), col_ref=df.columns[0], col_loop=df.columns[1::])
# df_nanfilter = utils.consecutive_measurements(df.reset_index(), col_object=df.columns[0], col_index='timestamp')

# Selecting the longest of them
idx_max = t_itv["Consecutive_Measurements"].astype(float).idxmax()
# Keep only largest consecutive time interval without NaNs
df_clean = df.loc[t_itv["BeginDate"][idx_max]: t_itv["EndDate"][idx_max]]

# Interpolate and drop columns with missing vals
df_clean.interpolate(method='time', inplace=True)
df_clean.dropna(axis=1, inplace=True)
buildings = df_clean.columns.tolist()


########################################################################################################################
print("Formating \n")
########################################################################################################################

df_dict = dict()
for blg in buildings:
    df_dict[blg] = df_clean.loc[:, [blg]]
    df_dict[blg].rename(columns={blg: 'elec'}, inplace=True)


########################################################################################################################
print(" \n Hierarchy initialization")
########################################################################################################################
### Creating spatial tree
root = treepkg.Node(id='A',
                    parent=None,
                    children=buildings)
nodes = {root}
for leaf_id in buildings:
    leaf = treepkg.Node(id=leaf_id,
                        parent='A',
                        children=None)
    nodes.add(leaf)
tree_S = treepkg.Tree(nodes, dimension='spatial')
# Creating data hierarchy
tree_S.create_spatial_hierarchy(df_dict, columns2aggr=['elec'])


X = tree_S.df_leaves('elec')

########################################################################################################################
print(" \n Mining: clustering")
########################################################################################################################

### Hierarchical clustering
Z = sch.linkage(X.T.values, method='ward', optimal_ordering=False)
leaves_labels = np.array(X.columns)
dendro = sch.dendrogram(Z,
                        #distance_sort='ascending',
                        truncate_mode='level',
                        p=0,
                        labels=leaves_labels,
                        get_leaves=True)
# Create tree class object from Z linkage matrix
S, y_ID = utils.create_Smatrix_from_linkagematrix(Z, leaves_labels)
tree = treepkg.Tree((S, y_ID), dimension='spatial')


cluster_thresholds = {'Bear': 35000,
                      'Bobcat': 15000,
                      'Bull': 170000,
                      'Cockatoo': 25000,
                      'Crow': 2500,
                      'Eagle': 30000,
                      'Fox': 40000,
                      'Gator': 22500,
                      'Hog': 80000,
                      'Lamb': 4000,
                      'Moose': 12000,
                      'Mouse': 5,
                      'Panther': 20000,
                      'Peacock': 12500,
                      'Rat': 41000,
                      'Robin': 27000,
                      'Shrew': 5,
                      'Swan': 4000,
                      'Wolf': 12000,
                      }

cluster_lengths = dict()
for site_file in site_files:
    site = site_file.split("\\")[8].split("_")[0]
    df = pd.read_csv(site_file, index_col=[0])
    cluster_lengths[site] = len(df.columns.tolist())

# cluster_lengths = {'Bear': 95, 'Bobcat': 31, 'Bull': 126,
#                    'Cockatoo': 118, 'Crow': 8, 'Eagle': 107,
#                    'Fox': 140, 'Gator': 74, 'Hog': 147, 'Lamb': 147,
#                    'Moose': 16, 'Mouse': 9, 'Panther': 108, 'Peacock': 46,
#                    'Rat': 294, 'Robin': 55, 'Shrew': 12, 'Swan': 22,
#                    'Wolf': 39}
