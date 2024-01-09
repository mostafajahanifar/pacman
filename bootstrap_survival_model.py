import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from lifelines import utils
from numpy import dot, einsum, log, exp, zeros, arange, multiply, ndarray

from survival_utils import CoxPH_train_val, CoxPH_train_infer, clinic_eval, extract_train_val_hazard_ratios

def normalize_datasets(train_set, test_set, feats_list, norm_type='meanstd'):
    if norm_type == 'meanstd':
        feat_mean = train_set[feats_list].mean()
        feat_std = train_set[feats_list].std()
        
        train_set_normalized = train_set.copy()
        test_set_normalized = test_set.copy()
        train_set_normalized[feats_list] =( train_set[feats_list] - feat_mean ) / feat_std
        test_set_normalized[feats_list] =( test_set[feats_list] - feat_mean ) / feat_std
    if norm_type == 'minmax':
        feat_min = train_set[feats_list].min()
        feat_max = train_set[feats_list].max()
        
        train_set_normalized = train_set.copy()
        test_set_normalized = test_set.copy()
        train_set_normalized[feats_list] =( train_set[feats_list] - feat_min ) / (feat_max - feat_min)
        test_set_normalized[feats_list] =( test_set[feats_list] - feat_min ) / (feat_max - feat_min)
    
    return train_set_normalized, test_set_normalized

def find_uncorrolated_features_old(data_df, feats_list, corr_thresh):
    corr = data_df[feats_list].corr().abs()
    upper_tri = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
    corrolated_features = [column for column in upper_tri.columns if any(upper_tri[column] > corr_thresh)]
    uncorr_feats_list = list(set(feats_list).difference(set(corrolated_features)))
    return uncorr_feats_list

def find_uncorrolated_features(dataset, feats_list, threshold):
    col_corr = set() # Set of all the names of deleted columns
    dataset = dataset[feats_list]
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return dataset.columns, col_corr
 
if __name__ == '__main__':
    ##path where to save the plots etc
    plots_save_path = 'all_feats_combi/KM_plots/'
    ##path to the csv file containing features for all the sets/cases including train/val
    discov_val_feats_path = 'all_feats_combi/features_combined/NOTT/discovery_valid_combined.csv'
    discov_df = pd.read_csv(discov_val_feats_path)
    all_feats_list = ['g1g3_cooccur', 'grade1n2_tilab_morisita', 'clusterCoff_mean', 'pleo_conf_60', 'tumor_immu_cell_cooccur', 'inter_WSI_T_density_correlation', 'grade2n3_morisita', 'grade_top_5', 'pleo_top_101', 'stil'] # list(discov_df.columns[223:310]) # Digital Features: [39:] --- mitotic features: [223:310] --- Clinical Features: [5:20]
    
    print('Number of features in the input: ', len(all_feats_list))
    
    ##### Feature cleanup: first based on std and then based on correlation (also manually based on previous experiments)
    std_thresh = 0 # normalized standard deviation
    corr_tresh = 1
    
    feature_exclusion_list = ['cenCloseness_min', 'cenHarmonic_min', 'nodeDegrees_min', 'LVI_per_regionpatch_30', 'grade2n3_tilab_shannon', 'tsarea_ratio_tarea', 'pleo1n2_tilab_shannon', 'pleo1n3_tilab_shannon']
    feats_list = []
    # remove useless features
    features_std = discov_df.std()
    for feat in all_feats_list:
        if features_std[feat] > 0 and feat not in feature_exclusion_list:
           feats_list.append(feat) 
           
    print('Number of features that have std larger than 0: ', len(feats_list))
    
    # remove features that have low variance in normalized values:
    normalized_df = (discov_df[feats_list] - discov_df[feats_list].min()) / (discov_df[feats_list].max() - discov_df[feats_list].min())
    features_std = normalized_df.std()
    for feat in feats_list.copy():
        if features_std[feat] < std_thresh:
            feats_list.remove(feat)
    print(f'Number of normalized features that have std larger than {std_thresh}: ', len(feats_list))
           
    # corrolation check
    # feats_list = find_uncorrolated_features_old(discov_df, feats_list, corr_tresh)
    feats_list, corrolated_feats_list = find_uncorrolated_features(discov_df, feats_list, corr_tresh)
    print('Number of features uncorrelated features: ', len(feats_list))
    
    # defining the parameters for survival analysis
    model_type = 'cox' ## 'rsf': for Random Survival Forest, 'cox': for Cox PH regression model
    save_plot = True ##whether to save the KM curve plots
    censor_at = 120 #in months. e.g 10 years = 120 months. 180 for 15 years, 240 for 20 years. Use -1 if no censoring is required 
    rsf_rseed = 100 ## random seed for Random Survival Forest
    cutoff_mode = 'median' ## 'median' | 'mean' ## the cut off point calculation for stratification of high vs low risk cases
    cutoff_point =  -1 ## 0.92 ##if set to -1 then median will be used as cut off. If set to any other positive value then cut_mode option will be ignores and the fixed cut off provided will be used for stratification of high vs low risk cases
    time_col = 'TTDM/ month' #'TTDM/ month' | 'Breast cancer specific survival/ month'
    event_col = 'Distant Metastasis' #'Distant Metastasis' | 'Survival Status'
    subset = 'Endocrine_LN0' ##'Endocrine'|'Endocrine_LN0'; 'Endocrine': Endocrine treated only with lymph node 0-3; 'Endocrine_LN0': Endocrine treated lymph node negative
    
    ######################################### Start the bootstraping
    # initialize the output lists
    bootsrap_num = 1000
    EE = discov_df[event_col].to_numpy()
    rng = np.random.RandomState()
    c_indices = []
    p_values = []
    for run in tqdm(range(bootsrap_num)):
        index_train = list(rng.choice(np.nonzero(EE==0)[0],size = len(EE)-np.sum(EE),replace=True))+list(rng.choice(np.nonzero(EE==1)[0],size = np.sum(EE),replace=True))
        index_test = list(set(range(len(EE))).difference(index_train))
        
        train_set = discov_df.iloc[index_train]
        test_set = discov_df.iloc[index_test]
        
        train_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)
        
        train_set, test_set = normalize_datasets(train_set, test_set, feats_list, norm_type='meanstd')
        
        output = extract_train_val_hazard_ratios(train_set, test_set, plots_save_path, feats_list, time_col, event_col, subset, censor_at, save_plot, cutoff_mode, cutoff_point)
        
        if output==-1 : #something went wrong, neglegct this run
            continue
        
        if run == 0:
            test_hazard_ratios, c_index, p_value = output
        else:
            temp, c_index, p_value = output
            test_hazard_ratios = pd.concat([test_hazard_ratios, temp], axis=1, join='outer', ignore_index=True, sort=False)
        c_indices.append(c_index)
        p_values.append(p_value)

    # hrs_mean = test_hazard_ratios.mean(axis=1)
    # hrs_std = test_hazard_ratios.std(axis=1)
    
    # print('Mean C-Index: ', np.mean(c_indices))
    # print('Std C-Index: ', np.std(c_indices))
    
    # # finding the top features based on stats
    # num_top_features = 20
    # temp = hrs_mean.sort_values(ascending=False)
    # top_features = temp.index.to_list()[:num_top_features]
    
    # # top features to dataframe
    # top_test_hazard_ratios = test_hazard_ratios.T[top_features]
    
    # print('Top features are:')
    # print(top_features)
    
    # top_test_hazard_ratios.to_csv(f'bootstrap_results_{event_col}_{subset}_stdThresh{std_thresh}_top20.csv')
    
    df_to_save =  test_hazard_ratios.T
    df_to_save['c_index'] = c_indices
    df_to_save['p_value'] = p_values
    df_to_save.to_csv(f'bootstrap_results_mitosisFeats_{event_col}_{subset}_stdThresh{std_thresh}_NEWcorrThresh{corr_tresh}.csv')
