import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import heapq
import statsmodels.api as sm

print("Loading utils...")
########################################################################################################################
###  Utility functions
########################################################################################################################

def flatten(list_of_lists):
    """"Source: https://stackabuse.com/python-how-to-flatten-list-of-lists/"""
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

def nested_dict_values_iterator(dict_obj, key_search):
    ''' This function accepts a nested dictionary as argument
        and iterate over all values of nested dictionaries until key_search is found
        Adapted from: https://thispointer.com/python-iterate-loop-over-all-nested-dictionary-values/
    '''
    # If value is dict then check if it has key_search
    if key_search in dict_obj.keys():
        yield dict_obj
    else:
        # Iterate over all values of given dictionary
        for value in dict_obj.values():
            # Check if value is of dict type
            if isinstance(value, dict):
                # If value is dict then iterate over all its values
                for v in nested_dict_values_iterator(value, key_search):
                    yield v
            else:
                # If value is not dict type then yield the value
                yield value


def add_element(dict, key, value):
    """Utility function for dictionary appending."""
    if key not in dict:
        dict[key] = []
    dict[key].append(value)


def dict_depth(dic: dict) -> int:
    """ Function to find the depth of a nested dictionary.
    Source: https://www.geeksforgeeks.org/python-find-depth-of-a-dictionary/"""
    if isinstance(dic, dict):
        return 1 + (max(map(dict_depth, dic.values()))
                    if dic else 0)
    return 1


def recursive_items(dictionary: dict):
    """ Function to access keys and values of a nested dictionary.
    Adapted from: https://stackoverflow.com/questions/39233973/get-all-keys-of-a-nested-dictionary"""
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key, None)
            yield from recursive_items(value)
        else:
            yield (key, value)


def recursive_items_all(dictionary: dict):
    """ Function to access keys and values of a nested dictionary.
    Source: https://stackoverflow.com/questions/39233973/get-all-keys-of-a-nested-dictionary"""
    for key, value in dictionary.items():
        if type(value) is list:
            yield (key, value)
        else:
            yield (key, value)
            yield from recursive_items_all(value)


def recursive_items_wdepth(dictionary: dict):
    """ Function to access keys, values & current depth of a nested dictionary."""
    for key, value in dictionary.items():
        current_depth = dict_depth(value)
        if type(value) is dict:
            yield (key, None, current_depth)
            yield from recursive_items_wdepth(value)
        else:
            yield (key, value, current_depth)


def keypath_from_nestedkeyval(dict, key_in):
    """"Function to return the dictionary key path from a nested key only"""
    all_key_list = []
    for key, value in recursive_items_all(dict):
        value_str = str(value)
        if str(key_in) in value_str:
            all_key_list.append(key)
        if key == key_in:
            return all_key_list

# Functions to iterate over nested dictionaries
# check - https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
from functools import reduce  # forward compatibility for Python 3
import operator


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def consecutive_measurements(df_in, col_object, col_index='timestamp'):
    """Function to count consecutive measurements and missing values within a dataframe column.
    Adapted from: https://stackoverflow.com/questions/26911851/how-to-use-pandas-to-find-consecutive-same-data-in-time-series.

    Parameters:
    df_in:      dataframe object
    col_object: string object pointing to the column of interest
    col_index:  string object pointing to a datatime column (typially a dataframe index set as a column)"""

    df_manip = df_in.copy()
    # Counting NaN values as int where 1 = NaN, and 0 = measurement
    df_manip["Nan_int"] = df_manip[col_object].isna().astype('int')

    # Obtaining the cumulative counts of values
    df_manip['value_grp'] = (df_manip["Nan_int"].diff(1) != 0).astype('int').cumsum()
    # Grouping them into a frame and extraacting begin and end Dates of cumulative sequences
    df_nanfilter = pd.DataFrame({'BeginDate': df_manip.groupby('value_grp')[col_index].first(),
                                 'EndDate': df_manip.groupby('value_grp')[col_index].last(),
                                 'Consecutive': df_manip.groupby('value_grp').size(),
                                 'Val': df_manip.groupby('value_grp').Nan_int.first()})
    return df_nanfilter


def inner_intersect(df_in,
                    col_ref: str,
                    col_loop: list):
    """This function returns the inner intersection of consecutive measurements between columns of an input dataframe"""

    df_nanfilter = consecutive_measurements(df_in, col_object=col_ref, col_index='timestamp')
    df_nanfilter = consecutive_missingval_thershold(df_nanfilter, threshold=8)
    df_measurements1 = df_nanfilter[df_nanfilter["Val"] == 0]

    t_itv = pd.DataFrame(columns=["BeginDate", "EndDate", "Consecutive_Measurements"])
    for index1, row1 in df_measurements1.iterrows():
        empty_intersect = False
        t_start_max, t_end_min = row1["BeginDate"], row1["EndDate"]
        # print('Main Loop Row1 : ' +str(row1["BeginDate"]))

        for col in col_loop:
            df_nanfilter2 = consecutive_measurements(df_in, col_object=col, col_index='timestamp')
            df_nanfilter2 = consecutive_missingval_thershold(df_nanfilter2, threshold=4)
            df_measurements2 = df_nanfilter2[df_nanfilter2["Val"] == 0]

            for index2, row2 in df_measurements2.iterrows():
                t_start = max(row1["BeginDate"], row2["BeginDate"])
                t_end = min(row1["EndDate"], row2["EndDate"])

                if t_start > t_end:  # no intersect
                    empty_intersect = True
                    pass
                else:
                    empty_intersect = False
                    t_start_max = max(t_start, t_start_max)
                    t_end_min = max(t_end, t_end_min)
                    pass

                if empty_intersect == False:
                    consecutive_val = float(len(pd.date_range(start=t_start_max, end=t_end_min, freq='15T')))
                    df_to_concat = pd.DataFrame({"BeginDate": t_start_max, "EndDate": t_end_min,
                                                 "Consecutive_Measurements": consecutive_val}, index=[0])
                    t_itv = pd.concat([t_itv, df_to_concat],
                                      ignore_index=True)

    return t_itv.drop_duplicates().reset_index(drop=True)


def consecutive_missingval_thershold(df_nanfilter,
                                     threshold: int = 4):
    """Adding a threshold for consecutive missing value consideration"""

    df_nanfilter_collapsed = df_nanfilter.copy()
    # Looping over the missing values below the given threshold
    for index, row in df_nanfilter.iterrows():

        # Skip first and last row if missing vals are obeserved
        if index == 1 and row["Val"] == 1:
            continue
        elif index == len(df_nanfilter) and row["Val"] == 1:
            continue

        # Otherwise proceed to collapse the dataframe
        elif row["Val"] == 1 and row["Consecutive"] < threshold:
            identifying_moving_index = df_nanfilter_collapsed["EndDate"] == df_nanfilter.loc[index-1, "EndDate"]
            index_in_collapsing_df = list(df_nanfilter_collapsed[identifying_moving_index].index)[0]

            # Extend the date of the measured data - index-1 row
            df_nanfilter_collapsed.loc[index_in_collapsing_df, "EndDate"] = df_nanfilter.loc[index + 1, "EndDate"]
            # Sum the consecutive values of missing & measured data
            sum_consecutive = df_nanfilter.loc[index - 1, "Consecutive"] + row.loc["Consecutive"] + df_nanfilter.loc[
                index + 1, "Consecutive"]
            df_nanfilter.loc[index - 1, "Consecutive"] = sum_consecutive
            # Collapse the frame
            df_nanfilter_collapsed.drop([index_in_collapsing_df + 1, index_in_collapsing_df + 2], inplace=True)
            df_nanfilter_collapsed.reset_index(drop=True, inplace=True)

    return df_nanfilter_collapsed


########################################################################################################################
### Feature Selection
########################################################################################################################
def extract_largest_pacf(pac_in: list,
                         ci: list,
                         n: int,
                         thresholdval: float):
    """Extracts the n-largest partial auto-correlation (pac) values
    from a given (pac_in) list and confidence interval (ci).
    A threshold can be set under which pac values will not be retained."""
    pac_in_ci = []
    for i, pac in enumerate(pac_in):
        if (pac < ci[i][0]/2 or pac > ci[i][1]/2) and np.abs(pac) >= thresholdval:
            pac_in_ci.append(pac)  # keep on the pac values outside the confidence interval
    pac_largest = heapq.nlargest(n, pac_in_ci[1::])  # do not keep the 1st value of the pac (autocorrelation)
    pac_largest_indices = [i for i in range(len(pac_in)) if pac_in[i] in pac_largest]
    if len(pac_largest) == 0:
        new_threshold = max(np.abs(pac_in[1::]))
        print("No partial auto-correlation was found respecting the defined threshold. Threshold is re-adapted to highest PAC value: ", new_threshold)
        pac_largest, pac_largest_indices = extract_largest_pacf(pac_in, ci, n, new_threshold)
    return pac_largest, pac_largest_indices


def get_pac_largest_timedeltas(df_in, 
                               n: int = 3, 
                               threshold: float = 0.25):
    pacf, ci = sm.tsa.pacf(df_in, alpha=0.05)
    # We extract the values and indices of the 3 largest partial auto-correlations
    pac_largest, pac_largest_indices = extract_largest_pacf(pacf, ci, n, threshold)
    # And obtain the time interval in pandas.Timedelta.isoformat()
    pac_largest_timedeltas = [df_in.index[i] - df_in.index[0]
                              for i in pac_largest_indices]
    pac_largest_timedeltas = [delta.isoformat() for delta in pac_largest_timedeltas]
    return pac_largest_timedeltas


########################################################################################################################
###  Clustering
########################################################################################################################

def find_leaves_from_Z(cluster_idx, Z, leaves=[]):
    n_samples = np.shape(Z)[0] + 1
    children_tomerge = Z[cluster_idx][0:2]

    if all(children_tomerge < n_samples):  # tree leaves
        leaves.append(list(children_tomerge))
    elif all(children_tomerge >= n_samples):  # both elements are nodes
        for child in children_tomerge:
            find_leaves_from_Z(int(child - n_samples), Z, leaves)
    else:
        leaves.append(children_tomerge.min())
        find_leaves_from_Z(int(children_tomerge.max()-n_samples), Z, leaves)
    leaves = flatten(leaves)
    leaves = [int(leaf) for leaf in leaves]
    return leaves


def generate_node_keys(n_samples, node_key_ensemble=[]):
    ascii_uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    if len(ascii_uppercase) >= n_samples or len(node_key_ensemble) == 0:
        node_key_ensemble = [key for key in ascii_uppercase[0:n_samples]]
    else:
        node_key_ensemble = [key1 + key2 for key1 in ascii_uppercase for key2 in node_key_ensemble]

    if len(node_key_ensemble) < n_samples:
        node_key_ensemble = generate_node_keys(n_samples, node_key_ensemble)

    # cut out extra values
    node_key_ensemble = node_key_ensemble[0:n_samples]
    return node_key_ensemble


def create_Smatrix_from_linkagematrix(Z: np.array, leaves_labels):
    n_samples = np.shape(Z)[0]+1
    node_key_ensemble = generate_node_keys(n_samples)

    S = np.zeros([n_samples*2-1, n_samples])
    y_ID = [''] * (n_samples-1)

    for i, items in enumerate(Z[::-1]):
        child1, child2, distance, count = items
        children_tomerge = np.array([child1, child2])

        if all(children_tomerge < n_samples):  # tree leaf
            cluster_idx = [int(child) for child in children_tomerge]
        elif all(children_tomerge >= n_samples):  # both elements are nodes
            cluster_idx = find_leaves_from_Z(int(children_tomerge[0]) - n_samples, Z, leaves=[])
            cluster_idx2 = find_leaves_from_Z(int(children_tomerge[1]) - n_samples, Z, leaves=[])
            cluster_idx = cluster_idx + cluster_idx2
        else:  # one element is a node, and one is a leaf
            cluster_idx = find_leaves_from_Z(int(children_tomerge.max()) - n_samples, Z, leaves=[])
            cluster_idx.append(int(children_tomerge.min()))

        # append summation matrix and y vector
        for j in cluster_idx:
            S[i, j] = 1
        y_ID[i] = node_key_ensemble[i]

    # add identity matrix at the bottom of the S matrix
    S[n_samples-1: n_samples*2-1] = np.identity(n_samples)
    y_ID = y_ID + list(leaves_labels)
    return S, y_ID


def cluster_Smatrix_from_threshold(Z: np.array, S: np.array, y_ID: list, threshold: int):

    indexes_to_delete = []
    for i, items in enumerate(Z[::-1]):
        child1, child2, distance, count = items

        if distance < threshold:
            indexes_to_delete.append(i)

    S_clustered = np.delete(S, indexes_to_delete, axis=0)
    y_ID_clustered = [y_id for i, y_id in enumerate(y_ID) if i not in indexes_to_delete]

    return S_clustered, y_ID_clustered


########################################################################################################################
###  Visualization
########################################################################################################################
hfont = {'fontname': 'Times New Roman'}

# source: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", **hfont)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.yticks(**hfont)
    plt.xticks(**hfont)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.4g}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            if valfmt(data[i, j], None) == '--':
                text = im.axes.text(j, i, '', **kw)
            else:
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


print("done!")
