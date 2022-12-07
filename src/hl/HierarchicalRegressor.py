########################################################################################################################
#                                               Hierarchical Regressor                                                 #
#                                                                                                                      #
#  This file implements the hierarchical regressor class along with some pre-mining related functions. The class takes #
# as input a Tree object defined in the TreeClass.py file. The hierarchical regressor is mainly responsible for feature#
# engineering, data partitioning, covariance matrix calculations, regressor definition, and result saving.             #
# The Base regressor is implemented at the very end of the file.                                                       #
########################################################################################################################

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras.models as km
import keras.backend as K

import numpy as np
import pandas as pd
import math
from scipy.linalg import block_diag
from scipy.stats import gmean
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy

from typing import Tuple
import pickle
from keras.models import clone_model
from os.path import exists

import hierarchicalearning.utils as utils
import hierarchicalearning.TreeClass as tc

########################################################################################################################
### PreMining utilities
########################################################################################################################

def add_time_features(df_in: pd.DataFrame,
                      trigo_transf: bool = True):
    """ Function to add time features to the input dataframe.

    trigo_transf       - bool - boolean operator defining whether or not to create a trigonometric transform of the
                                time index."""

    df_new = copy.deepcopy(df_in)

    if trigo_transf:
        df_new["Hour_Sin"] = np.sin(df_new.index.hour*(2. * np.pi / 24))
        df_new["Hour_Cos"] = np.cos(df_new.index.hour*(2. * np.pi / 24))
        df_new['DoY_Sin'] = np.sin(df_new.index.dayofyear * (2. * np.pi / 365))
        df_new['DoY_Cos'] = np.cos(df_new.index.dayofyear*(2. * np.pi / 365))
        # df_new['Month_Sin'] = np.sin(df_new.index.month * (2. * np.pi / 12))
        # df_new['Month_Cos'] = np.cos(df_new.index.month * (2. * np.pi / 12))
        df_new['DoW_Sin'] = np.sin(df_new.index.dayofweek * (2. * np.pi / 7))
        df_new['DoW_Cos'] = np.cos(df_new.index.dayofweek * (2. * np.pi / 7))
    else:
        df_new["Hour"] = df_new.index.hour
        df_new['DoY'] = df_new.index.dayofyear
        df_new['DoW'] = df_new.index.dayofweek
        #df_new['Month'] = df_new.index.month

    return df_new


def add_weather_features(df_in: pd.DataFrame, df_weather: pd.DataFrame):
    """ Function to add weather features to the input dataframe."""
    df_new = copy.deepcopy(df_in)
    df_new = pd.concat([df_new, df_weather])
    return df_new

def scale_df_columns(df_in, target_columns: list, scaler=StandardScaler()):
    """"Scaling dataframe per target_columns (list) column, returning a dataframe"""
    # Initialization
    nontarget_columns = list(set(df_in.columns) - set(target_columns))
    df = copy.deepcopy(df_in[target_columns])
    # Scaling
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_scaled[df_in.index.name] = df_in.index
    df_scaled.set_index([df_in.index.name], inplace=True, drop=True)
    # Adding nontarget_columns back to the dataframe for output
    df_scaled[nontarget_columns] = df_in[nontarget_columns]
    return df_scaled



########################################################################################################################
### Hierarchical Forecasting
########################################################################################################################

class H_Regressor():
    """A Hierarchical regressor class object to encapsulate hierarchical predictive learning related functions."""

    def __init__(self, tree_obj: tc.Tree,
                       target_in: list,
                       features_in: dict,
                       forecasting_method: str,
                       reconciliation_method: str,
                       **kwargs) -> None:

        self.tree = tree_obj
        self.targets = {key: target_in for key in self.tree.y_ID}
        self.target = target_in
        self.features = features_in
        self.input_length = sum([len(self.features[n]) for n in self.features])
        self.output_length = self.tree.n
        self.scale_divider = None
        self.scale_shift = None
        #self.input_with_only_leaves = None
        self.hierarchical_forecasting_methods = ['base', 'multitask', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV',
                                                 'kCOV']
        self.hierarchical_reconciliation_methods = ['None', 'OLS', 'BU', 'STR', 'hVAR', 'sVAR', 'COV', 'kCOV']
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.5
        if forecasting_method not in self.hierarchical_forecasting_methods:
            raise ValueError('Invalid hierarchical forecasting method. Possible methods are : base, multitask, OLS, '
                             'BU, STR, hVAR, sVAR, COV, kCOV')
        else:
            self.hierarchical_forecasting_method = forecasting_method
        if reconciliation_method not in self.hierarchical_reconciliation_methods:
            raise ValueError('Invalid reconciliation method. Possible methods are : None, OLS, BU, STR, hVAR, sVAR, COV'
                             ', kCOV')
        else:
            self.reconciliation_method = reconciliation_method

    def _to_klvl_block(self, M, **kwargs):
        """Enforces matrix M to be a block matrix according to k_level structure"""

        block_dim = kwargs['dimension'] if 'dimension' in kwargs else None
        # If the input tree is uni-dimensional, convert M to block format the following way
        if self.tree.dimension != 'spatiotemporal':
            # Looping over tree k_levels
            for aggr_lvl in self.tree.k_level_map.values():
                # Identifying nodes belonging to the same k-aggregation level
                node_map = [1 if id in aggr_lvl else 0 for i, id in enumerate(self.tree.y_ID)]
                # Fixing to 0 the elements not belonging to that block in the M matrix
                for i, elem in enumerate(node_map):
                    if elem == 1:
                        M[i, :] = M[i, :]*node_map
                        M[:, i] = M[:, i]*node_map
        elif block_dim is not None:
            # If the input tree is spatio-temporal & kwarg dimension is specified, convert M to block format along the
            # dimensional block specified
            if block_dim == 'temporal':
                k_lvl_vals_considered = self.tree.temporal.k_level_map.values()
                id_i = 1
            elif block_dim == 'spatial':
                k_lvl_vals_considered = self.tree.spatial.k_level_map.values()
                id_i = 0
            for T_aggr_lvl in k_lvl_vals_considered:
                # Map the nodes in that multi-level of aggregation
                bin_nodes_aggrlvl = [1 if id[id_i] in T_aggr_lvl else 0 for i, id in enumerate(self.tree.y_ID)]
                # Fixing to 0 the elements not belonging to that block in the M matrix
                for i, elem in enumerate(bin_nodes_aggrlvl):
                    if elem == 1:
                        M[i, :] = M[i, :] * bin_nodes_aggrlvl
                        M[:, i] = M[:, i] * bin_nodes_aggrlvl
            # If the input tree is spatio-temporal, convert M to block format along both spatio & temporal blocks only
        elif self.tree.dimension == 'spatiotemporal' and block_dim is None:
            for S_aggr_lvl in self.tree.spatial.k_level_map.values():
                for T_aggr_lvl in self.tree.temporal.k_level_map.values():
                    # Map the nodes in that multi-level of aggregation
                    bin_nodes_aggrlvl = [1 if id[0] in S_aggr_lvl and id[1] in T_aggr_lvl else 0 for i, id in
                                         enumerate(self.tree.y_ID)]
                    # Fixing to 0 the elements not belonging to that block in the M matrix
                    for i, elem in enumerate(bin_nodes_aggrlvl):
                        if elem == 1:
                            M[i, :] = M[i, :] * bin_nodes_aggrlvl
                            M[:, i] = M[:, i] * bin_nodes_aggrlvl
        return M

    def _create_markov(self, y_diff):
        """Create the markov scaling block diagonal matrix"""
        idx_list = [0] + list(np.cumsum(self.tree.y_ID))
        rho = []
        for i, j in zip(idx_list[:-1], idx_list[1:]):
            tmp = np.squeeze(y_diff[:, i:j].reshape(-1, 1))
            rho.append(acf(tmp, nlags=1, fft=True)[1])  # acf -> statsmodels.tsa.stattools.acf
        blocks = []
        for k, i in enumerate(self.tree.y_ID):
            tmp = np.eye(i)
            for j in range(i):
                for w in range(i):
                    if j != w:
                        tmp[j, w] = rho[k] ** (abs(j - w))
            blocks.append(tmp)
        return block_diag(*blocks)  # block_diag -> scipy.linalg.block_diag

    def sigma_shrinkage_wrapper(self, sigma_in, R_in, Dhvar, reconciliation_method):
        """We shrink the correlation matrix for COV and kCOV reconciliation methods.
        The wrapper additionally verifies if the covariance matrix is singular and replaces it with the identity matrix
        if singular by design."""

        if np.linalg.det(sigma_in) == 0:
            if reconciliation_method not in ['COV', 'kCOV']:
                print("Warning the covariance matrix is Singular by design in method: " + str(reconciliation_method))
                sigma = np.eye(self.output_length)
                return sigma

        if reconciliation_method in ['COV', 'kCOV'] and R_in is not None:
            lda = self.calc_optimal_lambda(sigma_in)
            R_srk = self.correlation_shrinkage(R_in, lda=lda)
            sigma = np.matmul(np.matmul(Dhvar, R_srk), Dhvar)
        else:
            sigma = sigma_in

        # Full shrinkage if the above still produces a singular sigma matrix
        if np.linalg.det(sigma) == 0:
            print("Warning the covariance matrix is still Singular, proceeding to maximal shrinkage with lambda 1")
            R_maxsrk = self.correlation_shrinkage(R_in, lda=1)
            sigma = np.matmul(np.matmul(Dhvar, R_maxsrk), Dhvar)

        # Back to identity matrix if problem
        if np.linalg.det(sigma) == 0:
            print("Warning the covariance matrix is Singular by design after shrinkage")
            sigma = np.eye(self.output_length)

        return sigma

    def sigma_shrinkage(self, sigma_in, lda=None):
        """Covariance matrix shrinkage"""
        lda = 0.5 if lda is None else lda
        sigmashrink = np.multiply(lda * np.eye(self.tree.n), sigma_in) + (1 - lda) * sigma_in
        return sigmashrink

    def correlation_shrinkage(self, corr_in, lda=None):
        """Correlation matrix shrinkage"""
        lda = 0.5 if lda is None else lda
        corrshrink = lda * np.eye(self.tree.n) + (1 - lda) * corr_in
        return corrshrink

    def calc_optimal_lambda(self, sigma_in):
        """Computes the optimal estimated shrinkage intensity from a given covariance (sigma) input matrix"""
        cov_sigma = np.cov(sigma_in, bias=True)
        # Forcing sigma & cov_sigma diagonal elements to zero
        for i in range(np.shape(sigma_in)[0]):
            sigma_in[i, i] = 0
            cov_sigma[i, i] = 0
        sigma_elem_sum = np.sum(sigma_in ** 2)
        cov_sigma_elem_sum = np.sum(cov_sigma)
        try:
            # Obtaining optimal lambda coefficient
            optimal_lambda = cov_sigma_elem_sum / sigma_elem_sum
        except ZeroDivisionError:
            optimal_lambda = 1
        if optimal_lambda > 1 or optimal_lambda < 0:
            optimal_lambda = 1
        return optimal_lambda

    def coherency_constraint(self, y_diff=None, **kwargs):
        """Calculating the covariance matrix from a given reconciliation method."""
        reconciliation_method = kwargs['method'] if 'method' in kwargs else self.hierarchical_forecasting_method #self.reconciliation_method
        reconciliation_method_in = 'None' if reconciliation_method == 'multitask' else reconciliation_method
        output = kwargs['output'] if 'output' in kwargs else False

        if reconciliation_method == 'BU':  # bottom up
            # Initializing the G matrix with zeros
            G = np.zeros((self.tree.m, self.tree.n))
            for i, elem in enumerate(self.tree.leaves_label):
                G[i, :] = [1 if id == elem else 0 for id in self.tree.y_ID]
            SG = np.matmul(self.tree.S, G)

            if output:
                return SG
            else:
                self.SG = SG
                self.SG_K = K.variable(value=self.SG)
                return
        else:
            sigma = self.covariance_matrix_calc(reconciliation_method_in, y_diff)
            sigma_inv = np.linalg.inv(sigma)
            SsigmaS = np.matmul(np.matmul(self.tree.S.T, sigma_inv), self.tree.S)
            SG = np.matmul(self.tree.S, np.matmul(np.matmul(np.linalg.inv(SsigmaS), self.tree.S.T), sigma_inv))
            if output:
                return SG
            else:
                self.sigma = sigma
                self.sigma_inv = sigma_inv
                self.SG = SG
                self.SG_K = K.variable(value=self.SG)

    def covariance_matrix_calc(self, reconciliation_method, y_diff_in=None):
        """Calculating the covariance matrix from a given reconciliation method."""

        R, Dhvar = None, None

        if y_diff_in is None:
            # Ordinary least squares default
            sigma = np.eye(self.output_length)
        else:
            if reconciliation_method == 'OLS':  # ordinary least squares
                sigma = np.eye(self.output_length)

            elif reconciliation_method == 'STR':  # structurally weighted least squares (WLS)
                leaves_maping = [1 if id in self.tree.leaves_label else 0 for i, id in enumerate(self.tree.y_ID)]
                leaves_var = np.var(y_diff_in[leaves_maping])
                sigma = leaves_var*np.diag(np.sum(self.tree.S, axis=1))

            elif reconciliation_method == 'hVAR':  # heterogeneous variance WLS
                if np.shape(y_diff_in)[1] == 1:
                    print("Warning: not enough samples to get a forecast error variance estimate per node")
                    var = np.var(y_diff_in)
                else:
                    var = np.var(y_diff_in, axis=1)
                sigma = np.diag(var)

            # homogeonous variance WLS uni-dimensional
            elif reconciliation_method == 'sVAR' and self.tree.dimension != 'spatiotemporal':
                svar = np.zeros(self.tree.n)
                for aggr_lvl in self.tree.k_level_map.values():
                    # Map the nodes in and not in that level of aggregation
                    map_nodes_aggrlvl = [i for i, id in enumerate(self.tree.y_ID) if id in aggr_lvl]
                    map_nodes_notinaggrlvl = [i for i, id in enumerate(self.tree.y_ID) if id not in aggr_lvl]
                    # Drop the rows not belonging to that level of aggregation
                    y_diff_aggrlvl = np.delete(y_diff_in, map_nodes_notinaggrlvl, axis=0)
                    # Estimate the common variance amongst this (homogenous) level of aggregation
                    node_var = np.var(y_diff_aggrlvl)
                    # Append the svar vector
                    svar[map_nodes_aggrlvl] = node_var
                sigma = np.diag(svar)

            # homogeonous variance WLS multi-dimensional
            elif reconciliation_method == 'sVAR' and self.tree.dimension == 'spatiotemporal':
                svar = np.zeros(self.tree.n)
                for S_aggr_lvl in self.tree.spatial.k_level_map.values():
                    for T_aggr_lvl in self.tree.temporal.k_level_map.values():
                        # Map the nodes in and not in that level of aggregation
                        map_nodes_aggrlvl = [i for i, id in enumerate(self.tree.y_ID) if
                                             id[0] in S_aggr_lvl and id[1] in T_aggr_lvl]
                        map_nodes_notinaggrlvl = [i for i, id in enumerate(self.tree.y_ID) if
                                                  i not in map_nodes_aggrlvl]
                        # Drop the rows not belonging to that level of aggregation
                        y_diff_aggrlvl = np.delete(y_diff_in, map_nodes_notinaggrlvl, axis=0)
                        # Estimate the common variance amongst this (homogenous) level of aggregation
                        node_var = np.var(y_diff_aggrlvl)
                        # Append the svar vector
                        svar[map_nodes_aggrlvl] = node_var
                sigma = np.diag(svar)

            elif reconciliation_method == 'COV':  # full covariance weighted least squares
                hvar = np.sqrt(np.var(y_diff_in, axis=1))
                Dhvar = np.diag(hvar)
                R = np.corrcoef(y_diff_in)
                R = np.nan_to_num(R, copy=True, nan=0.0, posinf=None, neginf=None)
                sigma = np.matmul(np.matmul(Dhvar, R), Dhvar)
                #sigma = np.cov(y_diff, bias=True)  # equivalent

            elif reconciliation_method == 'kCOV':  # k-level covariance weighted least squares
                hvar = np.sqrt(np.var(y_diff_in, axis=1))
                Dhvar = np.diag(hvar)
                R = self._to_klvl_block(np.corrcoef(y_diff_in))
                R = np.nan_to_num(R, copy=True, nan=0.0, posinf=None, neginf=None)
                sigma = np.matmul(np.matmul(Dhvar, R), Dhvar)
                #sigma = self._to_klvl_block(np.cov(y_diff, bias=True))  # equivalent

            elif reconciliation_method == 'multitask' or reconciliation_method == 'None':
                R, Dhvar = None, None
                sigma = np.eye(self.output_length)
            else:
                print('The reconciliation method has not been well defined. Review class initialization to avoid this '
                      'message.')
                R, Dhvar = None, None
                sigma = np.eye(self.output_length)

        # We verify if the covariance matrix is singular and proceed to shrink it if it is
        sigma = self.sigma_shrinkage_wrapper(sigma, R, Dhvar, reconciliation_method)
        return sigma

    def scaler_attribute_update(self, divider, shift):
        self.scale_divider = K.variable(value=divider)
        self.scale_shift = K.variable(value=shift)

    def coherency_loss_function_mse(self, y_true, y_hat):
        # Accuracy loss - mean squared Error (MSE)
        accuracy_loss_mse = K.mean(K.square(y_true - y_hat), axis=-1)
        # Coherency loss - MSE
        y_hat_x = K.transpose(y_hat) * self.scale_divider + self.scale_shift
        SGyhat = K.dot(self.SG_K, y_hat_x) / self.scale_divider - self.scale_shift / self.scale_divider
        coherency_loss = K.transpose(y_hat) - SGyhat
        #coherency_loss = K.transpose(y_true) - SGyhat
        coherency_loss_mse = K.mean(K.square(coherency_loss), axis=0)
        # Sum of both loss functions
        loss = accuracy_loss_mse*self.alpha + K.transpose(coherency_loss_mse)*(1-self.alpha)
        return loss

    def independent_loss_function_mse(self, y_true, y_hat):
        # Accuracy loss - Mean Squared squared Error (MSE)
        accuracy_loss_mse = K.mean(K.square(y_true - y_hat), axis=-1)
        return accuracy_loss_mse

    def coherency_loss_function_ms3e(self, y_true, y_hat):
        # Accuracy loss - mean structurally scaled squared Error (MS3E)
        accuracy_loss_ms3e = K.mean(K.square((y_true - y_hat)/self.tree.k_level_y[None, :]), axis=-1)
        # Coherency loss - str-MSE
        SGyhat = K.dot(self.SG_K, K.transpose(y_hat))
        coherency_loss = K.transpose(y_hat) - SGyhat
        coherency_loss_ms3e = K.mean(K.square(coherency_loss/self.tree.k_level_y[:, None]), axis=0)
        # Sum of both loss functions
        loss = accuracy_loss_ms3e*self.alpha + K.transpose(coherency_loss_ms3e)*(1-self.alpha)
        return loss
    
    def independent_loss_function_ms3e(self, y_true, y_hat):
        # Accuracy loss - Mean Squared scaled squared Error (MS3E)
        accuracy_loss_ms3e = K.mean(K.square((y_true - y_hat)/self.tree.k_level_y[None, :]), axis=-1)
        return accuracy_loss_ms3e

    def create_ANN_network(self, number_of_layer=3):
        # kernel_initializerialising the ANN
        ANNRegressor = Sequential()
        # Adding the input layer and the first hidden layer
        ANNRegressor.add(Dense(units=self.input_length,
                               kernel_initializer='lecun_uniform',
                               activation='sigmoid',
                               input_dim=self.input_length))
        ANNRegressor.add(Dropout(0.2))
        for i in range(1, number_of_layer):
            # Adding the i-th hidden layer
            ANNRegressor.add(Dense(units=self.input_length
                                         - round((self.input_length - self.output_length) * (i / number_of_layer)),
                                   kernel_initializer='lecun_uniform',
                                   activation='sigmoid'))
            ANNRegressor.add(Dropout(0.2))
        # Adding the last hidden layer
        ANNRegressor.add(Dense(units=self.output_length,
                               kernel_initializer='lecun_uniform',
                               activation='linear'))
        # Compiling the neural network
        if self.hierarchical_forecasting_method != 'multitask':
            ANNRegressor.compile(loss=self.coherency_loss_function_mse,
                                 optimizer='Adam',
                                 metrics=['mae', 'mse'])
        else:
            ANNRegressor.compile(loss=self.independent_loss_function_mse,
                                 optimizer='Adam',
                                 metrics=['mae', 'mse'])
        self.regressor = ANNRegressor

    def yhat_initialization(self, columns_in=None):
        columns = self.tree.hcolumns if columns_in is None else columns_in
        tree_df = self.tree.create_empty_hierarchy(columns_in=columns)
        self.yhat = tree_df

    def ytild_initialization(self, columns_in=None):
        columns = self.tree.hcolumns if columns_in is None else columns_in
        tree_df = self.tree.create_empty_hierarchy(columns_in=columns)
        self.ytild = tree_df

    def adapt_feature_shift(self, feature_toshift: dict, n: str):
        """Feature shift adaptation function to ensure no data leakage within the temporal trees from temporal shifts
        that might overlap between nodes.

        feature_toshift - dict  - contains features as keys with lists of temporal shifts as items.
                                e.g. {'feature1': ['1H', 1D], 'feature2': [15T]}
        n               - tree node considered for the feature engineering."""

        adapted_feature_toshift = copy.deepcopy(feature_toshift)

        if self.tree.dimension != 'spatial':
            node_sampling_rate = pd.Timedelta(n.split('_')[0]) if self.tree.dimension == 'temporal' else \
                pd.Timedelta(n[1].split('_')[0])
            node_temporal_horizon = int(n.split('_')[1]) if self.tree.dimension == 'temporal' else \
                int(n[1].split('_')[1])
            node_temporal_shift = node_sampling_rate * node_temporal_horizon

            for key in feature_toshift:
                for shift_request in feature_toshift[key]:
                    shift_request_is_within_tree_time_horizon = pd.Timedelta(shift_request) <= node_temporal_shift
                    if shift_request_is_within_tree_time_horizon:
                        adapted_feature_toshift[key].remove(shift_request)
                if len(adapted_feature_toshift[key]) == 0:
                    del adapted_feature_toshift[key]
        return adapted_feature_toshift

    def feature_engineering_wrapper(self, feature_toshift: dict,
                                    all_spatial_nodes=False,
                                    all_temporal_nodes=True,
                                    auto_adapt_shifts=True):
        """Feature engineering function wrapper.

        feature_toshift   - dict - contains node or temporal k_lvl aggregation information as keys with lists of
                                   temporal shifts as items,
                                   {temporal_klvl1: {'feature1': ['P0DT12H0M0S', 'P1DT0H0M0S'],
                                                     'feature2': ['P0DT6H0M0S']},
                                   temporal_klvl2: {'feature2': ['P0DT12H0M0S']}}

        all_nodes         - bool - defines whether all tree nodes are considered for feature engineering or only leaves.
        auto_adapt_shifts - bool - defines whether feature shifts should be automatically adapted to the temporal frame
                                   of the tree or not. This should always be set to true for hierarchical forecasting,
                                   is only set to False for Base forecast case studies."""

        if self.tree.dimension == 'temporal':
            klvls_to_loopover = self.tree.k_level_map.items() if all_temporal_nodes else [(1, self.tree.k_level_map[1])]
            for klvl, nodes_toloopover in klvls_to_loopover:
                time_features = True if klvl == self.tree.K or self.hierarchical_forecasting_method == 'base' else False
                for n in nodes_toloopover:
                    print('engineering features of node: ', n)
                    self.feature_engineering(feature_toshift[klvl], n, auto_adapt_shifts,
                                             add_time_feat=time_features)

        elif self.tree.dimension == 'spatial':
            nodes_to_loopover = self.tree.y_ID if all_spatial_nodes else self.tree.leaves_label
            time_features = True
            for n in nodes_to_loopover:
                print('engineering features of node: ', n)
                self.feature_engineering(feature_toshift[n], n, auto_adapt_shifts, add_time_feat=time_features)
                time_features = False if self.hierarchical_forecasting_method != 'base' else True

        elif self.tree.dimension == 'spatiotemporal':
            klvls_to_loopover = self.tree.temporal.k_level_map.items() \
                if all_temporal_nodes else [(1, self.tree.temporal.k_level_map[1])]
            for klvl, temp_nodes in klvls_to_loopover:
                time_features = True if klvl == self.tree.K or self.hierarchical_forecasting_method == 'base' else False
                # nodes_considered = self.tree.y_ID if all_spatial_nodes else self.tree.leaves_label
                # nodes_toloopover = [n for n in nodes_considered if n[1] in temp_nodes]
                spatial_nodes_considered = self.tree.spatial.y_ID if all_spatial_nodes else self.tree.spatial.leaves_label
                nodes_toloopover = [(n_spatial, n_temp) for n_spatial in spatial_nodes_considered for n_temp in
                                    temp_nodes]
                for n in nodes_toloopover:
                    print('engineering features of node: ', n)
                    self.feature_engineering(feature_toshift[(n[0], klvl)], n, auto_adapt_shifts,
                                             add_time_feat=time_features)
                    time_features = False if self.hierarchical_forecasting_method != 'base' else True

        # Finaly update the input shape format of the class with the new updated set of features
        self.input_length = sum([len(self.features[key]) for key in self.features])
        # And readjust the index over the tree
        self.adjust_index_over_tree()

    def feature_engineering(self, feature_toshift: dict,
                            n: str,
                            auto_adapt_shifts: bool,
                            add_time_feat: bool = False):
        """Hierarchical tree features are shifted tailored to the tree format,
         and trigonometrical time transformations are added as input features across the tree.
        This ensures no data leakage within the temporal trees from temporal shifts that might overlap between nodes.

        feature_toshift - dict  - contains features as keys with lists of temporal shifts as items.
                                e.g. {'feature1': ['1H', 1D], 'feature2': [15T]}
        n               - tree node considered for the feature engineering."""

        adapted_feature_toshift = self.adapt_feature_shift(feature_toshift, n) if auto_adapt_shifts else feature_toshift

        if add_time_feat:
            self.tree.df[n] = add_time_features(self.tree.df[n], trigo_transf=True)

        self.tree.df[n] = self.feature_shift(self.tree.df[n],
                                             feature_shift_dict=adapted_feature_toshift)

        self.features[n] = [value for value in list(self.tree.df[n].columns)
                            if value not in list(feature_toshift.keys())]

    def split_scaled(self, nb_splits: int = 5, **kwargs) -> Tuple[list, list, list, list, dict]:
        """Method to obtain testing and training sets out of the hierarchical dataframe. The sets can be scaled such
        that the scaler is fitted on the training sets to avoid data leakage to the testing set."""

        # Optional boolean kwargs
        toscale = kwargs['toscale'] if 'toscale' in kwargs else True
        only_leaves_X = kwargs['only_leaves_X'] if 'only_leaves_X' in kwargs else \
            True if self.tree.dimension == 'spatial' else False
        #self.input_with_only_leaves = only_leaves_X
        scaler = kwargs['scaler'] if 'scaler' in kwargs else StandardScaler()

        root_id = self.tree.root.id  # extract index from root node of the tree
        # defining the maximum train set size in function fo the number of splits and dataframe length
        max_train_size = np.round(self.tree.df[root_id].shape[0] / (nb_splits + 1))
        # calling the TimeSeriesSplit function
        tscv = TimeSeriesSplit(n_splits=nb_splits, max_train_size=int(max_train_size))

        # Identifying training and testing indexes
        train_idx, test_idx = [], []
        # Creating testing & training sets
        X_train, X_test, y_train, y_test = [], [], [], []
        y_test_allflats, i = {}, 0
        # Saving target scaler information
        target_scalers = dict()
        target_scalers['shift'], target_scalers['divider'] = [], []

        tree_df_sc = copy.deepcopy(self.tree.df)

        # Splitting testing & training sets (for all nodes)
        for train_index, test_index in tscv.split(self.tree.df[root_id]):
            print('Splitting and scaling training/testing sets, iteration ', i, ' over ', nb_splits)
            train_idx.append(list(self.tree.df[root_id].iloc[train_index].index.values))
            test_idx.append(list(self.tree.df[root_id].iloc[test_index].index.values))
            idx_train_flat = utils.flatten(train_idx)

            if toscale:
                y_target_divider, y_target_shift = [], []
                # Scaling features per training set
                nodes_to_loop_over = self.tree.leaves_label if only_leaves_X else list(self.features.keys())  #self.tree.y_ID
                for l in nodes_to_loop_over:
                    tree_df_sc[l], _, _ = self.scale_df(self.tree.df[l], target_columns=self.features[l],
                                                        fit_index=idx_train_flat, scaler=scaler)
                # Scaling targets per training set
                for n in self.tree.y_ID:
                    tree_df_sc[n][self.targets[n]], divider, shift = self.scale_df(self.tree.df[n][self.targets[n]],
                                                                                   fit_index=idx_train_flat,
                                                                                   scaler=scaler,
                                                                                   get_target_scaler=True)
                    y_target_divider.append(divider)
                    y_target_shift.append(shift)
                target_scalers['divider'].append(np.transpose(y_target_divider))
                target_scalers['shift'].append(np.transpose(y_target_shift))

            X_train.append(self.tree.reshape_tree2flat(t_index_in=train_idx[-1], columns_in=self.features,
                                                       all_outputs=False, only_leaves=only_leaves_X,
                                                       tree_df_in=tree_df_sc))
            y_train.append(self.tree.reshape_tree2flat(t_index_in=train_idx[-1], columns_in=self.targets,
                                                       all_outputs=False, only_leaves=False, tree_df_in=tree_df_sc))
            X_test.append(self.tree.reshape_tree2flat(t_index_in=test_idx[-1], columns_in=self.features,
                                                      all_outputs=False, only_leaves=only_leaves_X,
                                                      tree_df_in=tree_df_sc))
            y_test.append(self.tree.reshape_tree2flat(t_index_in=test_idx[-1], columns_in=self.targets,
                                                      all_outputs=False, only_leaves=False, tree_df_in=tree_df_sc))
            # Saving all tree2flat outputs for reshaping in flat2tree after hierarchical forecasting
            y_test_allflats[i] = self.tree.reshape_tree2flat(t_index_in=test_idx[-1], columns_in=self.targets,
                                                             all_outputs=True, only_leaves=False,
                                                             tree_df_in=tree_df_sc)
            i += 1
        # Saving node id of X train input vector elements
        X_train_allflats = self.tree.reshape_tree2flat(t_index_in=train_idx[-1], columns_in=self.features,
                                                       all_outputs=True, only_leaves=only_leaves_X,
                                                       tree_df_in=tree_df_sc)
        self.X_id = X_train_allflats[1][0]

        self.target_scalers = target_scalers
        return X_train, X_test, y_train, y_test, y_test_allflats

    def scale_df(self,
                 df_in: pd.DataFrame,
                 target_columns: list = None,
                 fit_index: list = None,
                 scaler=StandardScaler(),
                 get_target_scaler=False):
        """Scaling dataframe fitted on the suggested fit_index and target_columns (list) column, returning a
        dataframe. The rest of the dataframe index is transformed by the scaler fitted on the fit_index."""

        # Initialization
        target_columns = target_columns if target_columns is not None else df_in.columns

        if len(target_columns) == 0:
            return df_in, None, None
        else:
            nontarget_columns = list(set(df_in.columns) - set(target_columns))
            df = copy.deepcopy(df_in[target_columns])

            # Adjusting index to leaf sampling frequency if temporal or spatiotemporal tree
            fit_index = fit_index if fit_index is not None else list(df_in.index)
            pandas_fit_index = pd.DataFrame(index=fit_index)
            if self.tree.dimension == 'temporal' or self.tree.dimension == 'spatiotemporal':
                # identify temporal shift between the reference input root-level fit_index and the index of df_in (node)
                shift = df.index[0] - pandas_fit_index.index[0]
                # then shift the input (root) index to obtain the appropriate node (shifted) index
                fit_index = pandas_fit_index.shift(1, freq=shift).index
                transform_index = list(set(df.index) - set(fit_index))
            elif self.tree.dimension == 'spatial':
                transform_index = list(set(df_in.index) - set(pandas_fit_index.index))

            # Scaling
            if get_target_scaler:
                scale_divider, scale_shift = self.get_target_scaler(df_in.loc[fit_index], scaler_in=scaler)
            else:
                scale_divider, scale_shift = None, None
            # transform fit on the fit_index
            df_scaled_fit = pd.DataFrame(scaler.fit_transform(df.loc[fit_index]), columns=df.columns, index=fit_index)
            df.loc[fit_index] = df_scaled_fit
            # transform on the transform_index
            df_scaled_transf = pd.DataFrame(scaler.transform(df.loc[transform_index]), columns=df.columns,
                                            index=transform_index)
            df.loc[transform_index] = df_scaled_transf

            # Adding nontarget_columns back to the dataframe for output
            df[nontarget_columns] = df_in[nontarget_columns]

            return df, scale_divider, scale_shift

    def get_target_scaler(self, df, scaler_in):
        """Saving regressor target scalers per node id"""
        if scaler_in.__str__() == 'MinMaxScaler()':
            divider = df[self.target].max() - df[self.target].min()
            shift = df[self.target].min()
        elif scaler_in.__str__() == 'StandardScaler()':
            divider = df[self.target].std()
            shift = df[self.target].mean()
        else:
            divider = 1
            shift = 0
        return divider, shift

    def inverse_scale_y(self, y_z):
        y_x = np.transpose(y_z) * self.scale_divider + self.scale_shift
        return np.transpose(y_x)

    def feature_shift(self,
                      df_in: pd.DataFrame,
                      feature_shift_dict: dict):
        """ Function to shift features of a dataframe given a dictionary of lists.

        feature_shift_dict - dict - defines the dataframe features that should be shifted. Dictionary keys are the
                                    features and its respective values are the shifts (time frequencies list) to
                                    consider.
                                    eg: {'feature1': ['2H', '1D'], 'feature2': ['15T']}."""

        tree_df = copy.deepcopy(self.tree.df)
        df_new = copy.deepcopy(df_in)
        ref_node_id = [key for key in self.tree.df if self.tree.df[key].equals(df_new)][0]
        for feat in feature_shift_dict:
            for fshift in feature_shift_dict[feat]:

                if self.tree.dimension == 'spatial':
                    df_new[feat + '_' + str(fshift)] = df_new[feat].shift(freq=pd.Timedelta(fshift))

                else:
                    treet = self.tree if self.tree.dimension == 'temporal' else self.tree.temporal
                    shift_ratio = fshift /treet.horizon
                    temporal_ref_node_id = ref_node_id if self.tree.dimension == 'temporal' else ref_node_id[1]
                    # Temporal k_lvl information
                    k_lvl = [key for key in treet.k_level_map if temporal_ref_node_id in treet.k_level_map[key]][0]
                    k_lvl_len = len(treet.k_level_map[k_lvl])

                    # If the shift ratio is a whole number, the targeted shifted data is located within the same node
                    if shift_ratio.is_integer():
                        df_new[feat + '_' + str(fshift)] = df_new[feat].shift(periods=int(shift_ratio))

                    # Otherwise, the targeted shifted data is localized based on the indicated shift
                    else:
                        klvl_time_delta = pd.Timedelta(treet.k_level_map[k_lvl][0].split('_')[0])
                        number_of_shifts = fshift / klvl_time_delta

                        if number_of_shifts < 1:
                            raise ValueError(
                                "Requested feature shift is below the smallest tree sampling rate. Please adapt "
                                "input requested shift.")
                        else:
                            target_node_shift_identification = k_lvl_len - int(number_of_shifts % k_lvl_len)
                            shifts_in_target_node = math.ceil(number_of_shifts / k_lvl_len)
                            if self.tree.dimension == 'temporal':
                                target_node_id = [node for node in self.tree.k_level_map[k_lvl]
                                                  if node.split('_')[1] == str(target_node_shift_identification)][0]
                            else:
                                # Full tree k_lvl information
                                k_lvl_full = [key for key in self.tree.k_level_map
                                         if ref_node_id in self.tree.k_level_map[key]][0]
                                target_node_id = [node for node in self.tree.k_level_map[k_lvl_full]
                                                  if node[1].split('_')[1] == str(target_node_shift_identification)
                                                  and ref_node_id[0] == node[0]][0]
                            df_new[feat + '_' + str(fshift)] = pd.Series(tree_df[target_node_id][feat].shift(
                                periods=shifts_in_target_node)).values
        return df_new

    def adjust_index_over_tree(self):
        """Function to adjust the size of indexes over the entire tree"""
        min_len_index = len(self.tree.df[self.tree.y_ID[0]].index)
        for n in self.tree.y_ID:
            self.tree.df[n].dropna(axis='index', inplace=True)
            min_len_index = min(min_len_index, len(self.tree.df[n].index))
        for n in self.tree.y_ID:
            len_current_index = len(self.tree.df[n].index)
            difference_to_be_dropped = len_current_index - min_len_index
            self.tree.df[n].drop(self.tree.df[n].index[:difference_to_be_dropped], inplace=True)

    def hard_ctr_reconciliation(self, y_diff: list, y_hat, **kwargs) -> np.array:
        """Reconciles the input vector y_hat according to a given (reconciliation) method.
        The input vector is considered unscaled for the reconciliation, i.e., scale_S=False.
        If the method is not declared the reconciliation method of the class is employed.
        A reconciliation method None returns the input y_hat vector as is.
        The input vector y must satisfy the ordering of the y_ID and be of shape (*,n) and return an array of the same
        shape."""
        reconciliation_method = kwargs['method'] if 'method' in kwargs else self.reconciliation_method
        if reconciliation_method == 'None':
            return y_hat
        elif reconciliation_method == 'BU':
            y_tild = np.matmul(self.SG, np.transpose(y_hat))
            return np.transpose(y_tild)
        else:
            output_value = True
            SG = self.coherency_constraint(y_diff=y_diff, method=reconciliation_method,
                                           output=output_value)
            y_tild = np.matmul(SG, np.transpose(y_hat))
            return np.transpose(y_tild)

    def get_coherency_score(self, y: np.array, metric='MS3E') -> np.ndarray:
        """A function to return the coherency score of a given y vector.
        The coherency score is calculated based on either the RMS3E (root mean structurally scaled squared error) or
        MSE (mean squared error) of the difference between the input vector y and its reconciliated self SGy.
        The input vector y must satisfy the ordering of the y_ID and be of shape (n,*).
        The coherency score returned is of shape (n,*)"""
        SGy = np.matmul(self.SG, y)
        if metric == 'MSE':
            coh_score = mean_squared_error(y.T, SGy.T, multioutput='raw_values')
        elif metric == 'MS3E':
            coh_score = np.mean(np.square((y.T - SGy.T) / self.tree.k_level_y[None, :]), axis=0)
        return coh_score

    def save_model(self, filename: str):
        """Function to save the class and all its attributes as filename.txt.
        The regressor cannot be saved to pickle files as deep-ML models are too large to be stored through pickle at the
         moment.
         To move around this we consequently copy the class and remove the regressor attribute to it may be saved in
         pickle, while we save the regressor separately using the keras package."""
        # Copying the regressor then 'removing' it from the class attributes
        model_copy = clone_model(self.regressor)
        self.regressor = None
        # Saving the class with pickle
        fileobject = open(filename + '_class.txt', 'wb', )
        pickle.dump(self, fileobject)
        fileobject.close()
        # Saving the regressor separately
        model_copy.save(filename + '_regressor')

    def load(self, filename: str, hierarchical_forecasting_method):
        """Function to load a saved class and its model."""
        # Load object
        filehandler = open(filename + '_class.txt', 'rb')
        class_object = pickle.load(filehandler)
        filehandler.close()
        # Load regressor model
        if hierarchical_forecasting_method not in ['base', 'multitask']:
            keras_model = km.load_model(filename + '_regressor',
                                        custom_objects={'loss': self.coherency_loss_function_mse})
        elif hierarchical_forecasting_method == 'multitask':
            keras_model = km.load_model(filename + '_regressor',
                                        custom_objects={'loss': self.independent_loss_function_mse})
        # Placing the loaded model as the attribute of the class_object
        class_object.regressor = keras_model
        return class_object

    def save_performance_metrics(self, filename: str, y_true: np.array, y_hat: np.array, comput_time, iteration,
                                 **kwargs):
        """Function to save the performance metrics of the regressor.
        The input vectors y_* must satisfy the ordering of the y_ID and be of shape (*,n)."""
        r_method = kwargs['reconciliation_method'] if 'reconciliation_method' in kwargs else self.reconciliation_method
        # Load or create the results dictionary
        if exists(filename):
            # Load file
            filehandler = open(filename, 'rb')
            dic_df_res = pickle.load(filehandler)
            filehandler.close()

        else:
            # Initialization of columns to save
            results_f_r_cols = [str(i) + '_' + str(j) for i in self.hierarchical_forecasting_methods for j in
                                self.hierarchical_reconciliation_methods]
            # Creating dictionary of dataframes of results
            dic_df_res = dict()
            global_perf_metrics = ['global accuracy MSE', 'global accuracy MS3E', 'coherency MSE', 'coherency MS3E',
                                   'computation', 'global_successful_forecast', 'global_y_hat_std', 'global_y_hat_mean']
            for key in global_perf_metrics:
                dic_df_res[key] = pd.DataFrame(columns=results_f_r_cols)

            pernode_perf_metrics = ['successful_forecast', 'y_hat_std', 'y_hat_mean', 'accuracy MSE', 'accuracy MS3E']
            for key in pernode_perf_metrics:
                dic_df_res[key] = dict()
                for col in results_f_r_cols:
                    dic_df_res[key][col] = pd.DataFrame()
        # Calculate the performance metrics of the given prediction and append the result dictionary
        column = str(self.hierarchical_forecasting_method) + '_' + str(r_method)

        # MSE accuracy performance
        y_mse = mean_squared_error(y_true, y_hat, multioutput='raw_values')  # mse per individual node
        dic_df_res['global accuracy MSE'].loc[iteration, column] = np.mean(y_mse)
        # MS3E accuracy performance
        y_ms3e = np.mean(np.square((y_true - y_hat)/self.tree.k_level_y[None, :]), axis=0)  # ms3e per node
        dic_df_res['global accuracy MS3E'].loc[iteration, column] = np.mean(y_ms3e)
        # MSE coherency performance
        y_coh_mse = self.get_coherency_score(y_hat.T, metric='MSE')
        dic_df_res['coherency MSE'].loc[iteration, column] = np.mean(y_coh_mse)
        # MS3E coherency performance
        y_coh_ms3e = self.get_coherency_score(y_hat.T, metric='MS3E')
        dic_df_res['coherency MS3E'].loc[iteration, column] = np.mean(y_coh_ms3e)
        # computation performance
        dic_df_res['computation'].loc[iteration, column] = comput_time

        # global y_hat characteristics
        dic_df_res['global_successful_forecast'].loc[iteration, column] = np.array([(i < 0).any() for i in y_hat]).any()
        dic_df_res['global_y_hat_std'].loc[iteration, column] = np.mean(np.std(y_hat, axis=0))
        dic_df_res['global_y_hat_mean'].loc[iteration, column] = np.mean(y_hat)
        # Save the result dictionary
        dic_df_res['m'] = self.tree.m
        # y_hat characteristics
        dic_df_res['successful_forecast'][column] = pd.concat([dic_df_res['successful_forecast'][column], pd.Series([(i < 0).any() for i in y_hat])])
        dic_df_res['y_hat_std'][column] = pd.concat([dic_df_res['y_hat_std'][column], pd.Series(np.std(y_hat, axis=0))])
        dic_df_res['y_hat_mean'][column] = pd.concat([dic_df_res['y_hat_mean'][column], pd.Series(np.mean(y_hat, axis=0))])
        # MSE
        dic_df_res['accuracy MSE'][column] = pd.concat([dic_df_res['accuracy MSE'][column], pd.Series(y_mse)])
        dic_df_res['accuracy MS3E'][column] = pd.concat([dic_df_res['accuracy MS3E'][column], pd.Series(y_ms3e)])

        # save dictionary to output folder
        fileobject = open(filename, 'wb', )
        pickle.dump(dic_df_res, fileobject)
        fileobject.close()


class Base_Regressor(H_Regressor):
    """A Base regressor class."""

    def __int__(self, tree_obj: tc.Tree,
                target_in: list,
                features_in: dict,
                reconciliation_method: str,
                forecasting_method: str,
                **kwargs) -> None:

        H_Regressor.__init__(self, tree_obj, target_in, features_in,
                             reconciliation_method,
                             forecasting_method,
                             **kwargs)

    def create_base_ANN_network(self, n, number_of_layer=3):
        # kernel_initializerialising the ANN
        ANNRegressor = Sequential()
        # Adding the input layer and the first hidden layer
        ANNRegressor.add(Dense(units=len(self.features[n]),
                               kernel_initializer='lecun_uniform',
                               activation='relu',
                               input_dim=len(self.features[n])))
        ANNRegressor.add(Dropout(0.2))
        # Adding the i-th hidden layer
        for i in range(1, number_of_layer):
            ANNRegressor.add(Dense(units=len(self.features[n]) - round((len(self.features[n]) - 1) * (i / number_of_layer)),
                                   kernel_initializer='lecun_uniform',
                                   activation='relu'))
            ANNRegressor.add(Dropout(0.2))
        # Adding the last hidden layer
        ANNRegressor.add(Dense(units=1,
                               kernel_initializer='lecun_uniform',
                               activation='linear'))
        # Compiling the neural network
        ANNRegressor.compile(loss='MSE',
                             optimizer='Adam',
                             metrics=['mae', 'mse'])
        return ANNRegressor

    def RMSE_loss_function(self, y_true, y_hat):
        # Accuracy loss - Root Mean Squared Error (RMSE)
        accuracy_loss_rmse = K.sqrt(K.mean(K.square((y_true - y_hat)), axis=-1))
        return accuracy_loss_rmse
