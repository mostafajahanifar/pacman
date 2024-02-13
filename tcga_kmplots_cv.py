import os
import glob
import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from survival_utils import normalize_datasets, plot_km, cross_validation_tcga

import argparse

BOOTSTRAP_RUNS = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--cancer_types', nargs='+', required=True)
    parser.add_argument('--cancer_subset', default=None)
    parser.add_argument('--event_type', required=True)
    parser.add_argument('--censor_at', type=int, default=-1)
    parser.add_argument('--results_root', default='./CV_results/')
    parser.add_argument('--splits_root', default=None) # 
    parser.add_argument('--cutoff_mode', default="optimal")

    # parser.add_argument('--cancer_types', nargs='+', default=['STAD'])
    # parser.add_argument('--cancer_subset', default=None)
    # parser.add_argument('--event_type', default='OS')
    # parser.add_argument('--censor_at', type=int, default=-1)
    # parser.add_argument('--results_root', default='./CV_results_NoCensor_optimal/')
    # parser.add_argument('--splits_root', default="/mnt/gpfs01/lsf-workspace/u2070124/Code/PORPOISE/splits/5foldcv/") # 
    # parser.add_argument('--cutoff_mode', default="median")

    args = parser.parse_args()

    cancer_types = args.cancer_types
    cancer_subset = args.cancer_subset
    event_type = args.event_type
    censor_at = args.censor_at
    results_root = args.results_root
    splits_root = args.splits_root
    cutoff_mode = args.cutoff_mode

    print(args)

    # defining the parameters for survival analysis
    nfolds = [1,2,3] # ALAKI
    model_type = 'cox' ## 'rsf': for Random Survival Forest, 'cox': for Cox PH regression model
    save_plot = True
    rsf_rseed = 100 ## random seed for Random Survival Forest
    cutoff_point =  -1 ## 0.92 ##if set to -1 then median will be used as cut off. If set to any other positive value then cut_mode option will be ignores and the fixed cut off provided will be used for stratification of high vs low risk cases
    time_col = f'{event_type}.time'
    event_col = event_type
    subset = cancer_subset
    
    # Raeading the data and filtering
    plots_save_path = 'all_feats_combi/KM_plots/' #ALAKI
    discov_val_feats_path = '/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_clinical_merged.csv'
    # discov_val_feats_path = '/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_clinical_merged.csv'
    discov_df = pd.read_csv(discov_val_feats_path)
    discov_df = discov_df.loc[discov_df['type'].isin(cancer_types)]
    discov_df = discov_df.dropna(subset=[event_col, time_col])
    discov_df[event_col] = discov_df[event_col].astype(int)
    discov_df[time_col] = (discov_df[time_col]/30.4).astype(int)
    print(f"Number of cases in FS experiment after dropping NA: {len(discov_df)}")

    # finding the list of best features
    FS_root = f'./FS_results/FS_results_10years/FS_{cancer_types}/'
    FS_path = FS_root + f'FS_{cancer_types}_{event_type}_censor{int(365*censor_at/12)}.txt'
    FS_df = pd.read_csv(FS_path, delimiter=';')
    FS_df = FS_df.sort_values(['c_index', 'p_value'], ascending=[False, True])
    best_feat_ind = FS_df.index[0]
    feats_list = ast.literal_eval(FS_df['selected_features'][best_feat_ind])

    # setting the results path
    save_dir = f'./{results_root}/CV_{cancer_types}/'
    os.makedirs(save_dir, exist_ok=True)

    if splits_root is not None:
        ex_iterator = range(5)
    else:
        ex_iterator = range(BOOTSTRAP_RUNS)

    # running the cross-validation here
    EE = discov_df[event_col].to_numpy()
    rng = np.random.RandomState()
    c_indices = []
    p_values = []
    test_cutoffs = []
    for run in tqdm(ex_iterator):
        # forming the training and validation cohorts
        if splits_root is None:
            index_train = list(rng.choice(np.nonzero(EE==0)[0],size = len(EE)-np.sum(EE),replace=True))+list(rng.choice(np.nonzero(EE==1)[0],size = np.sum(EE),replace=True))
            index_test = list(set(range(len(EE))).difference(index_train))
            train_set = discov_df.iloc[index_train]
            test_set = discov_df.iloc[index_test]
        else:
            split_folder = 'tcga_'+''.join(cancer_types).lower()
            split_path = splits_root+f'{split_folder}/splits_{run}.csv'
            split_df = pd.read_csv(split_path)
            train_set = discov_df[discov_df['bcr_patient_barcode'].isin(split_df['train'])]
            test_set = discov_df[discov_df['bcr_patient_barcode'].isin(split_df['val'])]
        train_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)

        print("Number of training samples: ", len(train_set), train_set[event_col].sum())
        print("Number of test_set samples: ", len(test_set), test_set[event_col].sum())
        print("Number of train_set + test_set samples: ", len(test_set)+len(train_set))

        # Normalizing the datasets
        train_set, test_set = normalize_datasets(train_set, test_set, feats_list, norm_type='meanstd')

        # running the model and collating the results
        output = cross_validation_tcga(train_set, test_set, plots_save_path, feats_list, time_col, event_col, subset, censor_at, save_plot, cutoff_mode, cutoff_point)
        if output==-1 : #something went wrong, neglegct this run
            print(f"something went wrong with run {run}")
            continue
        if run == 0:
            test_hazard_ratios, c_index, p_value, test_partial_hazard, test_data, test_cutoff, train_fig, val_fig = output
        if run != 0 and output != -1:
            temp, c_index, p_value, test_partial_hazard_, test_data_, test_cutoff, train_fig, val_fig = output
            test_hazard_ratios = pd.concat([test_hazard_ratios, temp], axis=1, join='outer', ignore_index=True, sort=False)
            test_partial_hazard = np.concatenate([test_partial_hazard, test_partial_hazard_], axis=0)
            test_data = pd.concat([test_data, test_data_], axis=0, join='outer', ignore_index=True, sort=False)
        test_cutoffs.append(test_cutoff)
        c_indices.append(c_index)
        p_values.append(p_value)

        folds_save_dir = save_dir + 'folds_results/'
        os.makedirs(folds_save_dir, exist_ok=True)
        save_fig_path = folds_save_dir + f'{cancer_types}_{event_type}_censor{censor_at}_cv{run}_train'
        train_fig.savefig(save_fig_path + '.png', dpi=600, bbox_inches = 'tight', pad_inches = 0)
        train_fig.savefig(save_fig_path + '.pdf', dpi=600, bbox_inches = 'tight', pad_inches = 0)

        save_fig_path = folds_save_dir + f'{cancer_types}_{event_type}_censor{censor_at}_cv{run}_val'
        val_fig.savefig(save_fig_path + '.png', dpi=600, bbox_inches = 'tight', pad_inches = 0 )
        val_fig.savefig(save_fig_path + '.pdf', dpi=600, bbox_inches = 'tight', pad_inches = 0)
    
    print("median p-value (corrected): ", 2*np.median(p_values))
    print("c-index (average): ", np.mean(c_indices))
    print("c-index (std): ", np.std(c_indices))
    df_to_save =  test_hazard_ratios.T
    df_to_save['c_index'] = c_indices
    df_to_save['p_value'] = p_values
    save_path = save_dir + f"cv_results_{cancer_types}_{event_type}_censor{censor_at}"
    df_to_save.to_csv(save_path+".csv", index=None)

    print("Evaluating the overall test set")
    cutoff_value = np.mean(test_cutoffs)
    overall_fig = plot_km(test_data[time_col], test_data[event_col], test_partial_hazard, cutoff_value, add_at_risk_counts=False, x_label="Months", y_label=f"{event_type} Probability")
    overall_fig.savefig(save_path + '.png', dpi=600, bbox_inches = 'tight', pad_inches = 0)
    overall_fig.savefig(save_path + '.pdf', dpi=600, bbox_inches = 'tight', pad_inches = 0)



    