import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import argparse


from survival_utils import CoxPH_train_val, clinic_eval, extract_train_val_hazard_ratios

import warnings
warnings.filterwarnings("ignore")

C_INDEX_THRESH = 0.6
BOOTSTRAP_RUNS = 100

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

    return list(dataset.columns), col_corr

## bootstraping
def bootstrap_cph(discov_df, EE, plots_save_path, feats_list, time_col, event_col, subset, nfolds, save_plot, model_type, censor_at, rsf_rseed, cutoff_mode, cutoff_point):
    rng = np.random.RandomState()
    c_indices = []
    p_values = []
    num_fails = 0
    for run in range(BOOTSTRAP_RUNS):
        index_train = list(rng.choice(np.nonzero(EE==0)[0],size = len(EE)-np.sum(EE),replace=True))+list(rng.choice(np.nonzero(EE==1)[0],size = np.sum(EE),replace=True))
        index_test = list(set(range(len(EE))).difference(index_train))
        
        train_set = discov_df.iloc[index_train]
        test_set = discov_df.iloc[index_test]
        
        train_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)
        
        train_set, test_set = normalize_datasets(train_set, test_set, feats_list, norm_type='meanstd')
        
        output = extract_train_val_hazard_ratios(train_set, test_set, plots_save_path, feats_list, time_col, event_col, subset, censor_at, save_plot, cutoff_mode, cutoff_point)
        
        if output==-1 : #something went wrong, neglegct this run
            num_fails += 1
            continue
        
        if run == 0:
            test_hazard_ratios, c_index, p_value = output
        else:
            temp, c_index, p_value = output
            test_hazard_ratios = pd.concat([test_hazard_ratios, temp], axis=1, join='outer', ignore_index=True, sort=False)
        c_indices.append(c_index)
        p_values.append(p_value)
    val_mean_Cindex = np.mean(c_indices)
    val_std_Cindex = np.std(c_indices)
    val_mean_Pvalue = 2*np.median(p_values)
    num_fails += np.count_nonzero(np.array(c_indices)<C_INDEX_THRESH)
    return val_mean_Cindex, val_std_Cindex, val_mean_Pvalue, num_fails

##cross validation on 3 different splits
def cross_val(splits_path, feats_path, plots_save_path, feats_list, time_col, event_col, subset, nfolds, save_plot, model_type, censor_at, rsf_rseed, cutoff_mode, cutoff_point):
    p_values_train = []
    c_indices_train = []
    hz_train = []
    hz_ci_low_train = []
    hz_ci_high_train = []

    p_values_test = []
    c_indices_test = []
    hz_test = []
    hz_ci_low_test = []
    hz_ci_high_test = []

    cut_points = []
        
    # print('feature(s): ', feats_list)

    hr_scores = np.zeros((649, 3))
    for k in nfolds:
        splits = pd.read_csv(splits_path)
        train_ids = splits['S' + str(k) + '_discov'].dropna()
        train_ids = pd.DataFrame(train_ids.to_list(), columns=['Case ID'])
        val_ids = splits['S' + str(k) + '_val'].dropna()
        val_ids = pd.DataFrame(val_ids.to_list(), columns=['Case ID'])

        feats = pd.read_csv(feats_path)
        train_set = feats.merge(train_ids, how='inner', on='Case ID')
        val_set = feats.merge(val_ids, how='inner', on='Case ID')

        if model_type == 'cox':                                             
            tpv, tcind, thz, thz_ci_l, thz_ci_h, vpv, vcind, vhz, vhz_ci_l, vhz_ci_h, cutpnt, hr_scores = CoxPH_train_val(train_set, val_set, plots_save_path, feats_list, time_col, event_col, subset, censor_at, save_plot, cutoff_mode, cutoff_point, hr_scores, k-1)
        elif model_type == 'clinic':
            tpv, tcind, thz, thz_ci_l, thz_ci_h, vpv, vcind, vhz, vhz_ci_l, vhz_ci_h, cutpnt, hr_scores = clinic_eval(train_set, val_set, plots_save_path, feats_list, time_col, event_col, subset, censor_at, save_plot, cutoff_mode, cutoff_point, hr_scores, k-1)
                  
        p_values_train.append(tpv)
        c_indices_train.append(tcind)
        hz_train.append(thz)
        hz_ci_low_train.append(thz_ci_l)
        hz_ci_high_train.append(thz_ci_h)

        p_values_test.append(vpv)
        c_indices_test.append(vcind)
        cut_points.append(cutpnt)
        hz_test.append(vhz)
        hz_ci_low_test.append(vhz_ci_l)
        hz_ci_high_test.append(vhz_ci_h)

    val_mean_Cindex = np.asarray(c_indices_test).mean()
    val_mean_Pvalue = 2*np.median(np.asarray(p_values_test))
    return val_mean_Cindex, val_mean_Pvalue



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cancer_types', nargs='+', required=True)
    parser.add_argument('--cancer_subset', default=None)
    parser.add_argument('--event_type', required=True)
    parser.add_argument('--censor_at', type=int, default=-1)
    parser.add_argument('--results_root', default='./FS_results/')

    args = parser.parse_args()

    cancer_types = args.cancer_types
    cancer_subset = args.cancer_subset
    event_type = args.event_type
    censor_at = args.censor_at
    results_root = args.results_root

    # num selected features
    num_features = 10
    
    ##path where to save the plots etc
    plots_save_path = 'all_feats_combi/KM_plots/' #ALAKI
    
    discov_val_feats_path = '/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_clinical_merged.csv'
    discov_df = pd.read_csv(discov_val_feats_path)
    discov_df = discov_df.loc[discov_df['type'].isin(cancer_types)]
    feats_list = pd.read_csv('noncorrolated_feature_list.csv', header=None)[0].to_list()

    print(f"Initial number of cases in this cancer type: {len(discov_df)}")
    
    nfolds = [1,2,3] # ALAKI
    model_type = 'cox' ## 'rsf': for Random Survival Forest, 'cox': for Cox PH regression model
    save_plot = False # ALAKI
    rsf_rseed = 100 ## random seed for Random Survival Forest
    cutoff_mode = 'median' ## 'median' | 'mean' ## the cut off point calculation for stratification of high vs low risk cases
    cutoff_point =  -1 ## 0.92 ##if set to -1 then median will be used as cut off. If set to any other positive value then cut_mode option will be ignores and the fixed cut off provided will be used for stratification of high vs low risk cases
    time_col = f'{event_type}.time'
    event_col = event_type
    subset = cancer_subset ##'Endocrine'|'Endocrine_LN0'; 'Endocrine': Endocrine treated only with lymph node 0-3; 'Endocrine_LN0': Endocrine treated lymph node negative
    
    discov_df = discov_df.dropna(subset=[event_col, time_col])
    discov_df[event_col] = discov_df[event_col].astype(int)
    discov_df[time_col] = discov_df[time_col].astype(int)

    print(f"Number of cases in FS experiment after dropping NA: {len(discov_df)}")
    
    EE = discov_df[event_col].to_numpy()
    save_dir = results_root + f"FS_{cancer_types}/"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"FS_{cancer_types}_{event_type}_censor{censor_at}.txt"
    fid = open(save_dir + save_name, 'w')
    fid.write('selected_features;c_index;c_index_std;p_value;num_fails\n')
    worst_score = 0
    selected_features = []
    # ranking_crits = [['c_index', 'p_value', 'num_fail'], ['p_value', 'c_index', 'num_fail']]
    for i in range(num_features):
        feature_scores = {'c_index': [], 'c_index_std': [], 'p_value': [], 'num_fail':[]}
        for fi in tqdm(range(len(feats_list))):
            f = feats_list[fi]
            temp_selected_features = selected_features.copy()
            if f in temp_selected_features:
                feature_scores['c_index'].append(0)
                feature_scores['c_index_std'].append(1000)
                feature_scores['p_value'].append(1000)
                feature_scores['num_fail'].append(BOOTSTRAP_RUNS+1)
                continue
            temp_selected_features.append(f)
            try:
                c_index, c_index_std, p_value, num_fail = bootstrap_cph(discov_df, EE, plots_save_path, temp_selected_features, time_col, event_col, subset, nfolds, save_plot, model_type, censor_at, rsf_rseed, cutoff_mode, cutoff_point)
                feature_scores['c_index'].append(c_index)
                feature_scores['c_index_std'].append(c_index_std)
                feature_scores['p_value'].append(p_value)
                feature_scores['num_fail'].append(num_fail)
            except:
                feature_scores['c_index'].append(0)
                feature_scores['c_index_std'].append(1000)
                feature_scores['p_value'].append(1000)
                feature_scores['num_fail'].append(BOOTSTRAP_RUNS+1)
        score_df = pd.DataFrame(feature_scores)
        score_df['scaled_c_index'] = (1-score_df.c_index_std) * score_df.c_index
        # score_df['final_score'] = (1-score_df.p_value) * score_df.c_index
        # score_df = score_df.sort_values(ranking_crits[i%2], ascending=[False, True, True])
        score_df = score_df.sort_values(['scaled_c_index', 'p_value', 'num_fail'], ascending=[False, True, True])
        best_feat_ind = score_df.index[0]
        selected_features.append(feats_list[best_feat_ind])
        print(f'Best features at step {i+1}/{num_features} are: {selected_features}')
        print(score_df.iloc[0])
        
        log = f'{selected_features};{score_df.iloc[0][0]};{score_df.iloc[0][1]};{score_df.iloc[0][2]};{score_df.iloc[0][3]}\n'
        fid.write(log)
        
        # shuffle the features for the next round
        random.shuffle(feats_list)
        
    fid.close()