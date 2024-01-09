import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter

from survival_utils import CoxPH_train_val, CoxPH_train_infer, clinic_eval

##script for fitting a survival model on different features contained in a csv file

def normalize_datasets(train_set, test_set, feats_list):
    feat_mean = train_set[feats_list].mean()
    feat_std = train_set[feats_list].std()
    
    train_set_normalized = train_set.copy()
    test_set_normalized = test_set.copy()
    train_set_normalized[feats_list] =( train_set[feats_list] - feat_mean ) / feat_std
    test_set_normalized[feats_list] =( test_set[feats_list] - feat_mean ) / feat_std
    
    return train_set_normalized, test_set_normalized

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
        
    print('feature(s): ', feats_list)

    hr_scores = np.zeros((649, 3))
    for k in nfolds:
        #path_to_train_set = main_path + '/fold' + str(k) + '/discov_3fold_' + str(k) + '.csv'
        #path_to_val_set = main_path + '/fold' + str(k) + '/valid_3fold_' + str(k) + '.csv'
        splits = pd.read_csv(splits_path)
        train_ids = splits['S' + str(k) + '_discov'].dropna()
        train_ids = pd.DataFrame(train_ids.to_list(), columns=['Case ID'])
        val_ids = splits['S' + str(k) + '_val'].dropna()
        val_ids = pd.DataFrame(val_ids.to_list(), columns=['Case ID'])

        feats = pd.read_csv(feats_path)
        train_set = feats.merge(train_ids, how='inner', on='Case ID')
        val_set = feats.merge(val_ids, how='inner', on='Case ID')
        
        # train_set, val_set = normalize_datasets(train_set, val_set, feats_list)

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

    print('Discovery mean c-index: ', np.asarray(c_indices_train).mean())
    print('Discovery sd c-index: ', np.asarray(c_indices_train).std())
    print('Validation mean c-index: ', np.asarray(c_indices_test).mean())
    print('Validation sd c-index: ', np.asarray(c_indices_test).std())
    print('Discovery mean p-value: ', 2*np.median(np.asarray(p_values_train)))
    print('Validation mean p-value: ', 2*np.median(np.asarray(p_values_test)))
    print('Discovery c-indeces: ', c_indices_train)
    print('Discovery p-values: ', p_values_train)
    print('Discovery HR: ', hz_train)
    print('Discovery HR_CI_low: ', hz_ci_low_train)
    print('Discovery HR_CI_high: ', hz_ci_high_train)
    print('cutoff mean: ', np.asarray(cut_points).mean())
        
    print('valid c-indeces: ', c_indices_test)
    print('Validation p-values: ', p_values_test)
    print('Validation HR: ', hz_test)
    print('Validation HR_CI_low: ', hz_ci_low_test)
    print('Validation HR_CI_high: ', hz_ci_high_test)
    print('Cutoffs: ', cut_points)
   
##do inference i-e generate Risk_score for the test set which can then be uploaded to ML evaluation server to get the evaluation done
def inference(splits_path, discov_val_path, test_path, plots_save_path, feats_list, time_col, event_col, subset, nfolds, save_plot, model_type, censor_at, rsf_rseed, cutoff_mode, cutoff_point):
    p_values_train = []
    c_indices_train = []
    hz_train = []
    hz_ci_low_train = []
    hz_ci_high_train = []

    cut_points = []
        
    print('feature(s): ', feats_list)

    hr_scores = np.zeros((649, 3))

    splits = pd.read_csv(splits_path)
    train_ids = splits['S' + str(nfolds) + '_discov'].dropna()
    train_ids = pd.DataFrame(train_ids.to_list(), columns=['Case ID'])
    val_ids = splits['test'].dropna()
    val_ids = pd.DataFrame(val_ids.to_list(), columns=['Case ID'])

    train_feats = pd.read_csv(discov_val_path)
    train_set = train_feats.merge(train_ids, how='inner', on='Case ID')
    val_feats = pd.read_csv(test_path)
    val_set = val_feats.merge(val_ids, how='inner', on='Case ID')
    
    # train_set, val_set = normalize_datasets(train_set, val_set, feats_list)
    
    #path_to_train_set = main_path + '/fold' + str(nfolds) + '/discov_3fold_' + str(nfolds) + '.csv'
    #path_to_val_set = test_path

    if model_type == 'cox':                                             
        tpv, tcind, thz, thz_ci_l, thz_ci_h, cutpnt, hr_scores = CoxPH_train_infer(train_set, val_set, plots_save_path, feats_list, time_col, event_col, subset, censor_at, save_plot, cutoff_mode, cutoff_point, hr_scores, nfolds)
                  
    p_values_train.append(tpv)
    c_indices_train.append(tcind)
    hz_train.append(thz)
    hz_ci_low_train.append(thz_ci_l)
    hz_ci_high_train.append(thz_ci_h)

    cut_points.append(cutpnt)
    
    print('Discovery c-indeces: ', c_indices_train)
    print('Discovery p-values: ', p_values_train)
    print('Discovery HR: ', hz_train)
    print('Discovery HR_CI_low: ', hz_ci_low_train)
    print('Discovery HR_CI_high: ', hz_ci_high_train)
    print('cutoff mean: ', np.asarray(cut_points).mean())

if __name__ == '__main__':
    ##path to the folder containing discovery and validation csv files for the 3 splits
    #main_path = '/data/data/NOTT/cell_surv/set1to18_5folds/exps/all_feats/kfolds/results/criteria2_L5_64_inceptionv3_mj_unet_cellcount_filter_train_mj_unet_wsi_patches_tumorfeats_regionpatch_allBRACE_3fold_new_v2/'
    splits_path = 'all_feats_combi/clinical_files/NOTT_splits.csv'
    
    ##path where to save the plots etc
    plots_save_path = 'all_feats_combi/KM_plots/'
    
    feats_list = ['mit_clusterCoff_mean', 'mit_cenEigen_h8', 'mit_cenDegree_h7', 'mit_cenKatz_h7'] # ['clusterCoff_mean', 'cenEigen_mean', 'cenCloseness_std', 'cenKatz_std', 'cenDegree_max', 'cenDegree_h3', 'cenEigen_max']##list of feature(s) to use for fitting a multivariate survival model
    # feats_list = [
    #               #mitosis related features
    #               'nodeDegrees_max', 'clusterCoff_mean', 'cenEigen_max', 'cenHarmonic_h1',# 'clusterCoff_mean',#,'assortCoeff',,'nodeDegrees_std','nodeDegrees_mean'
    #               # TILs related features
    #               'stil', 'stil_count','tcell_til',
    #               # stroma-tumour ratio related features
    #               'TSR', 'stroma_per_TS_regionpatch', 'inter_WSI_T_S_ratio_ASM',
    #               # heterogeneity features
    #               'inter_WSI_T_S_ratio_heterogeneity', 'inter_WSI_T_density_heterogeneity', 'intra_WSI_T_density_heterogeneity','overall_WSI_T_S_density_heterogeneity','overall_WSI_T_density_heterogeneity','overall_WSI_dcisT_density_heterogeneity',
    #               # TGCI
    #               'per_g1', 'per_g2', 'per_g3', 'tumor_per_all_regionpatch',
    #               # IHC features
    #               'CL_score_S', #'Perc_PT', 
    #               ] ##list of feature(s) to use for fitting a multivariate survival model
    
    # best p-value mitosis features
    # feats_list = ['cenEigen_h1', 'cenEigen_min', 'cenCloseness_h3', 'cenKatz_h5', 'cenDegree_h3']
    
    # #best c-index mitosis features
    # feats_list = ['cenHarmonic_h1', 'avrDegree_8']
    
    # # best based on bootstrap
    # feats_list = ['nodeDegrees_h7', 'clusterCoff_std', 'cenCloseness_max']
    
    # #Best clinical features
    # feats_list = ['M', 'LVI']
    
    ##path to the csv file containing features for all the sets/cases including train/val
    discov_val_feats_path = 'all_feats_combi/features_combined/NOTT/discovery_valid_combined_mitosisRefined.csv'
    
    nfolds = [1,2,3] ##number of splits to use for fitting the model and result generation. value from 1 to 3 like [1,2,3] to use all the 3 splits; [1,2] to use split 1 and 2; [3] only use split 3.
    model_type = 'cox' ## 'rsf': for Random Survival Forest, 'cox': for Cox PH regression model
    save_plot = True ##whether to save the KM curve plots
    censor_at = 120 #in months. e.g 10 years = 120 months. 180 for 15 years, 240 for 20 years. Use -1 if no censoring is required 
    rsf_rseed = 100 ## random seed for Random Survival Forest
    cutoff_mode = 'median' ## 'median' | 'mean' ## the cut off point calculation for stratification of high vs low risk cases
    cutoff_point =  -1## 0.92 ##if set to -1 then median will be used as cut off. If set to any other positive value then cut_mode option will be ignores and the fixed cut off provided will be used for stratification of high vs low risk cases

    time_col = 'TTDM/ month' #'TTDM/ month' | 'Breast cancer specific survival/ month'
    event_col = 'Distant Metastasis' #'Distant Metastasis' | 'Survival Status'

    subset = 'Endocrine_LN0' ##'Endocrine'|'Endocrine_LN0'; 'Endocrine': Endocrine treated only with lymph node 0-3; 'Endocrine_LN0': Endocrine treated lymph node negative
    
    ###### 3-splits cross validation ###@@@@@@@@@@@@@@@@@@@@@@@@@
    cross_val(splits_path, discov_val_feats_path, plots_save_path, feats_list, time_col, event_col, subset, nfolds, save_plot, model_type, censor_at, rsf_rseed, cutoff_mode, cutoff_point)

    ##########@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ inference @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ##use the best feature set and cut off point to generate risk_score for test set. This will be evaluated on ML evaluation server
    test_feats_path = 'all_feats_combi/features_combined/NOTT/NOTT_test_combined_mitosisRefined.csv'

    trn_fold = 2##which train fold to use for test inference
    #use this inference() for generating csv file with risk score for test cases which can then be evaluated on the ML eval server
    inference(splits_path, discov_val_feats_path, test_feats_path, plots_save_path, feats_list, time_col, event_col, subset, trn_fold, save_plot, model_type, censor_at, rsf_rseed, cutoff_mode, cutoff_point)

    ########@@@@@@@@@@@@@@@@@@@@@@@@@ clinical features evaluation  ##@@@@@@@@@@@@@@@@@@@@@@
    #model_type = 'clinic'
    #cross_val(splits_path, discov_val_feats_path, plots_save_path, feats_list, time_col, event_col, subset, nfolds, save_plot, model_type, censor_at, rsf_rseed, cutoff_mode, cutoff_point)

    print('Done')
