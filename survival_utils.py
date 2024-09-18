import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter
from sksurv.metrics import concordance_index_censored
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

def cross_validation_tcga_corrected(path_to_train_set, path_to_val_set, path_to_save_results, feats_list, time_col, event_col, subset, censor_at, plotit, cutoff_mode, cutoff_point):
    train_data = path_to_train_set
    # train_data = train_data.dropna()
    train_data = train_data.fillna(0)

    ##convert all events other than 1 to 0
    train_data.loc[train_data[event_col] > 1, event_col] = 0

    ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)

    ##these lines can be commented out (set 2 == 3 in the if) to train the model using lymph node 0-3 so that more event are included during training even though the validation might be done on LN- only.
    if subset == 'Endocrine_LN0' and 2 == 2:
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)
        train_data.drop(train_data[train_data['Lymph Node status'] == 1].index, inplace=True)
    
    feats_list_temp = []
    
    if isinstance(feats_list, str):
        feats_list_temp = [feats_list, time_col, 'event']
    else:
        for l in feats_list:
            feats_list_temp.append(l)
        feats_list_temp.append(time_col)
        feats_list_temp.append(event_col)

    train_data_temp = train_data[feats_list_temp]

    estim_method = 'breslow' ## 'breslow', 'spline', 'piecewise'
    l1_r = 0.5 # 0.5 ## specify what ratio to assign to a L1 vs L2 penalty.
    penalty = 0.001 # 0.001
    
    try:
        cph_mva = CoxPHFitter(baseline_estimation_method=estim_method, l1_ratio=l1_r, penalizer=penalty).fit(train_data_temp, duration_col=time_col, event_col=event_col)
    except:
        print('Failed in fitting CoxPH for train_data_infer')
        return -1

    #cph_mva.print_summary() ##print the summary of the fit #####@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    train_data_infer = train_data_temp.drop(columns=[event_col, time_col])
    partial_hazard_train = cph_mva.predict_partial_hazard(train_data_infer)
    
    # Use mean value in the discovery set as the cut-off value and divide subjects into two groups
    if cutoff_point == -1:
        if cutoff_mode == 'mean':
            cutoff_value = partial_hazard_train.mean()
        elif cutoff_mode == 'median':
            cutoff_value = partial_hazard_train.median()
        elif cutoff_mode == 'quantile':
            cutoff_value = partial_hazard_train.quantile(0.60)
        elif cutoff_mode == 'optimal':
            cutoff_value = find_optimal_threshold(partial_hazard_train, train_data_temp[time_col], train_data_temp[event_col])
        else:
            cutoff_value = cutoff_mode
    else:
        cutoff_value = cutoff_point
    
    ##################################################### Predict on the validation set
    val_data = path_to_val_set #pd.read_csv(path_to_val_set)
    #val_data = val_data.dropna()
    val_data = val_data.fillna(0)

    ##apply censoring. 
    if censor_at > 0:
        val_data.loc[val_data[time_col] > censor_at, event_col] = 0
        val_data.loc[val_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    val_data.loc[val_data[event_col] > 1, event_col] = 0

     ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        val_data.drop(val_data[val_data['Endocrine Therapy'] != 1].index, inplace=True)
        val_data.drop(val_data[val_data['Chemotherapy'] == 1].index, inplace=True)

    if subset == 'Endocrine_LN0':
        val_data.drop(val_data[val_data['Endocrine Therapy'] != 1].index, inplace=True)
        val_data.drop(val_data[val_data['Chemotherapy'] == 1].index, inplace=True)
        val_data.drop(val_data[val_data['Lymph Node status'] == 1].index, inplace=True)
    
    val_data_temp = val_data[feats_list_temp]

    partial_hazard_test = cph_mva.predict_partial_hazard(val_data_temp)
    # c-index on the validation set
    vcindex = concordance_index(val_data_temp[time_col], -partial_hazard_test, val_data_temp[event_col])

    # Use mean value in the discovery set as the cut-off value and divide subjects int the validation set into two groups
    upper = partial_hazard_test >= cutoff_value
    T_upper_test = val_data_temp[time_col][upper]
    E_upper_test = val_data_temp[event_col][upper]
    lower = partial_hazard_test < cutoff_value
    T_lower_test = val_data_temp[time_col][lower]
    E_lower_test = val_data_temp[event_col][lower]

    results = logrank_test(T_lower_test, T_upper_test, E_lower_test, E_upper_test)
    
    val_pvalue = results.p_value
    val_cindex = vcindex
    val_hr = cph_mva.hazard_ratios_

    return T_lower_test, T_upper_test, E_lower_test, E_upper_test, val_cindex, val_pvalue, val_hr


def cross_validation_tcga(path_to_train_set, path_to_val_set, path_to_save_results, feats_list, time_col, event_col, subset, censor_at, plotit, cutoff_mode, cutoff_point):
    train_data = path_to_train_set
    # train_data = train_data.dropna()
    train_data = train_data.fillna(0)

    ##apply censoring. Commenting these out as we can use the uncensored data for training but for validation will need to censor
    # if censor_at > 0:
    #     train_data.loc[train_data[time_col] > censor_at, event_col] = 0
    #     train_data.loc[train_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    train_data.loc[train_data[event_col] > 1, event_col] = 0

    ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)

    ##these lines can be commented out (set 2 == 3 in the if) to train the model using lymph node 0-3 so that more event are included during training even though the validation might be done on LN- only.
    if subset == 'Endocrine_LN0' and 2 == 2:
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)
        train_data.drop(train_data[train_data['Lymph Node status'] == 1].index, inplace=True)
    
    feats_list_temp = []
    
    if isinstance(feats_list, str):
        feats_list_temp = [feats_list, time_col, 'event']
    else:
        for l in feats_list:
            feats_list_temp.append(l)
        feats_list_temp.append(time_col)
        feats_list_temp.append(event_col)

    train_data_temp = train_data[feats_list_temp]

    estim_method = 'breslow' ## 'breslow', 'spline', 'piecewise'
    l1_r = 0.5 # 0.5 ## specify what ratio to assign to a L1 vs L2 penalty.
    penalty = 0.001 # 0.001
    
    try:
        cph_mva = CoxPHFitter(baseline_estimation_method=estim_method, l1_ratio=l1_r, penalizer=penalty).fit(train_data_temp, duration_col=time_col, event_col=event_col)
    except:
        print('Failed in fitting CoxPH for train_data_infer')
        return -1

    #cph_mva.print_summary() ##print the summary of the fit #####@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    train_data_infer = train_data_temp.drop(columns=[event_col, time_col])
    partial_hazard_train = cph_mva.predict_partial_hazard(train_data_infer)
    
    # Use mean value in the discovery set as the cut-off value and divide subjects into two groups
    if cutoff_point == -1:
        if cutoff_mode == 'mean':
            cutoff_value = partial_hazard_train.mean()
        elif cutoff_mode == 'median':
            cutoff_value = partial_hazard_train.median()
        elif cutoff_mode == 'quantile':
            cutoff_value = partial_hazard_train.quantile(0.60)
        elif cutoff_mode == 'optimal':
            cutoff_value = find_optimal_threshold(partial_hazard_train, train_data_temp[time_col], train_data_temp[event_col])
        else:
            cutoff_value = cutoff_mode
    else:
        cutoff_value = cutoff_point
    
    if plotit:
        train_fig = plot_km(train_data_temp[time_col],train_data_temp[event_col],partial_hazard_train,cutoff_value)
    ##################################################### Predict on the validation set
    val_data = path_to_val_set #pd.read_csv(path_to_val_set)
    #val_data = val_data.dropna()
    val_data = val_data.fillna(0)

    ##apply censoring. 
    if censor_at > 0:
        val_data.loc[val_data[time_col] > censor_at, event_col] = 0
        val_data.loc[val_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    val_data.loc[val_data[event_col] > 1, event_col] = 0

     ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        val_data.drop(val_data[val_data['Endocrine Therapy'] != 1].index, inplace=True)
        val_data.drop(val_data[val_data['Chemotherapy'] == 1].index, inplace=True)

    if subset == 'Endocrine_LN0':
        val_data.drop(val_data[val_data['Endocrine Therapy'] != 1].index, inplace=True)
        val_data.drop(val_data[val_data['Chemotherapy'] == 1].index, inplace=True)
        val_data.drop(val_data[val_data['Lymph Node status'] == 1].index, inplace=True)
    
    val_data_temp = val_data[feats_list_temp]

    partial_hazard_test = cph_mva.predict_partial_hazard(val_data_temp)
    # c-index on the validation set
    vcindex = concordance_index(val_data_temp[time_col], -partial_hazard_test, val_data_temp[event_col])

    # Use mean value in the discovery set as the cut-off value and divide subjects int the validation set into two groups
    upper = partial_hazard_test >= cutoff_value
    T_upper_test = val_data_temp[time_col][upper]
    E_upper_test = val_data_temp[event_col][upper]
    lower = partial_hazard_test < cutoff_value
    T_lower_test = val_data_temp[time_col][lower]
    E_lower_test = val_data_temp[event_col][lower]

    # Log-rank test: if there is any significant difference between the groups being compared
    results = logrank_test(T_lower_test, T_upper_test, E_lower_test, E_upper_test)
    
    val_pvalue = results.p_value
    val_cindex = vcindex
    val_hr = cph_mva.hazard_ratios_

    if plotit:
        val_fig = plot_km(val_data_temp[time_col],val_data_temp[event_col],partial_hazard_test,cutoff_value)
    
    if plotit:
        return val_hr, val_cindex, val_pvalue, partial_hazard_test, val_data_temp, cutoff_value, train_fig, val_fig
    return val_hr, val_cindex, val_pvalue, partial_hazard_test, val_data_temp, cutoff_value

def plot_km(time_df, event_df, partial_hazard_train, cutoff_value, add_at_risk_counts=True, x_label="Months", y_label=None):
    upper = partial_hazard_train >= cutoff_value
    T_upper_train = time_df[upper]
    E_upper_train = event_df[upper]
    lower = partial_hazard_train < cutoff_value
    T_lower_train = time_df[lower]
    E_lower_train = event_df[lower]

    # evaluating
    results = logrank_test(T_lower_train, T_upper_train, E_lower_train, E_upper_train)

    # preparing the figure
    font_size = 18
    fig_size = 10
    fig = plt.figure(figsize=(fig_size, fig_size-2)) ##adjust according to font size
    ax = fig.add_subplot(111)
    ax.set_xlabel('', fontsize=font_size)
    ax.set_ylabel('', fontsize=font_size)
        
    if results.p_value < 0.0001:
        train_pvalue_txt = 'p < 0.0001'
    else:
        train_pvalue_txt = 'p = ' + str(np.round(results.p_value, 4))

    from matplotlib.offsetbox import AnchoredText
    ax.add_artist(AnchoredText(train_pvalue_txt, loc=1, frameon=False, prop=dict(size=font_size)))
    ax.tick_params(axis='x', labelsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size)

    # Initializing the KaplanMeierModel for each group
    km_upper = KaplanMeierFitter()
    km_lower = KaplanMeierFitter()
    ax = km_upper.fit(T_upper_train, event_observed=E_upper_train, label='high').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='r', ci_show=False, xlabel=x_label, ylabel=y_label)
    ax = km_lower.fit(T_lower_train, event_observed=E_lower_train, label='low').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='b', ci_show=False, xlabel=x_label, ylabel=y_label)

    if add_at_risk_counts:
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(km_upper, km_lower, ax=ax, fig=fig, fontsize=int(font_size*1) )
        fig.subplots_adjust(bottom=0.4)
        fig.subplots_adjust(left=0.2)
    ax.get_legend().remove()
    #plt.show()
    return fig
        

def normalize_datasets(train_set, test_set, feats_list, norm_type='meanstd'):
    if norm_type == 'meanstd':
        feat_mean = train_set[feats_list].mean()
        feat_std = train_set[feats_list].std()
        
        train_set_normalized = train_set.copy()
        test_set_normalized = test_set.copy()
        train_set_normalized[feats_list] =( train_set[feats_list] - feat_mean ) / feat_std
        test_set_normalized[feats_list] =( test_set[feats_list] - feat_mean ) / feat_std
    if norm_type == 'meanstd_wrong':
        feat_mean = train_set[feats_list].mean()
        feat_std = train_set[feats_list].std()
        train_set_normalized = train_set.copy()
        train_set_normalized[feats_list] =( train_set[feats_list] - feat_mean ) / feat_std

        feat_mean = test_set[feats_list].mean()
        feat_std = test_set[feats_list].std()
        test_set_normalized = test_set.copy()
        test_set_normalized[feats_list] =( test_set[feats_list] - feat_mean ) / feat_std
    if norm_type == 'minmax':
        feat_min = train_set[feats_list].min()
        feat_max = train_set[feats_list].max()
        
        train_set_normalized = train_set.copy()
        test_set_normalized = test_set.copy()
        train_set_normalized[feats_list] =( train_set[feats_list] - feat_min ) / (feat_max - feat_min)
        test_set_normalized[feats_list] =( test_set[feats_list] - feat_min ) / (feat_max - feat_min)
    
    return train_set_normalized, test_set_normalized

def find_optimal_threshold(risk_factor, time_to_event, event_occurred):
    # Define the range of potential thresholds
    thresholds = np.linspace(risk_factor.min(), risk_factor.max(), 100)

    # Initialize the maximum log-rank statistic and the optimal threshold
    max_logrank_stat = 0
    min_p_value = 1000
    optimal_threshold = None

    # Iterate over the potential thresholds
    for threshold in thresholds:
        # Divide the data into groups based on the threshold
        groups = np.digitize(risk_factor, bins=[threshold])
        
        # Calculate the log-rank statistic
        result = logrank_test(time_to_event[groups == 0], time_to_event[groups == 1],
                            event_observed_A=event_occurred[groups == 0],
                            event_observed_B=event_occurred[groups == 1])
        
        # If this log-rank statistic is larger than the current maximum,
        # update the maximum and the optimal threshold
        if result.p_value < min_p_value:
            max_logrank_stat = result.test_statistic
            min_p_value = result.p_value
            optimal_threshold = threshold
    print(f'optimal threshold: {optimal_threshold} -> p-value: {min_p_value} , test_stat: {max_logrank_stat}')
    return optimal_threshold

def extract_train_val_hazard_ratios(path_to_train_set, path_to_val_set, path_to_save_results, feats_list, time_col, event_col, subset, censor_at, plotit, cutoff_mode, cutoff_point):
    train_data = path_to_train_set
    #train_data = train_data.dropna()
    train_data = train_data.fillna(0)

    ##apply censoring. Commenting these out as we can use the uncensored data for training but for validation will need to censor
    #train_data.loc[train_data[time_col] > censor_at, event_col] = 0
    #train_data.loc[train_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    train_data.loc[train_data[event_col] > 1, event_col] = 0

    ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        pass
    
    feats_list_temp = []
    
    if isinstance(feats_list, str):
        feats_list_temp = [feats_list, time_col, 'event']
    else:
        for l in feats_list:
            feats_list_temp.append(l)
        feats_list_temp.append(time_col)
        feats_list_temp.append(event_col)

    train_data_temp = train_data[feats_list_temp]

    estim_method = 'breslow' ## 'breslow', 'spline', 'piecewise'
    l1_r = 0.5 # 0.5 ## specify what ratio to assign to a L1 vs L2 penalty.
    penalty = 0.001 # 0.001
    
    try:
        cph_mva = CoxPHFitter(baseline_estimation_method=estim_method, l1_ratio=l1_r, penalizer=penalty).fit(train_data_temp, duration_col=time_col, event_col=event_col)
    except:
        print('Failed in fitting CoxPH for train_data_infer')
        return -1

    #cph_mva.print_summary() ##print the summary of the fit #####@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    train_data_infer = train_data_temp.drop(columns=[event_col, time_col])
    partial_hazard_train = cph_mva.predict_partial_hazard(train_data_infer)
    
    # Use mean value in the discovery set as the cut-off value and divide subjects into two groups
    if cutoff_point == -1:
        if cutoff_mode == 'mean':
            cutoff_value = partial_hazard_train.mean()
        elif cutoff_mode == 'median':
            cutoff_value = partial_hazard_train.median()
        elif cutoff_mode == 'quantile':
            cutoff_value = partial_hazard_train.quantile(0.60)
        elif cutoff_mode == 'optimal':
            cutoff_value = find_optimal_threshold(partial_hazard_train, train_data_temp[time_col], train_data_temp[event_col])
        else:
            cutoff_value = cutoff_mode
    else:
        cutoff_value = cutoff_point
    
    if plotit:
        train_fig = plot_km(train_data_temp[time_col],train_data_temp[event_col],partial_hazard_train,cutoff_value)
    ##################################################### Predict on the validation set
    val_data = path_to_val_set #pd.read_csv(path_to_val_set)
    #val_data = val_data.dropna()
    val_data = val_data.fillna(0)

    # ##apply censoring. 
    if censor_at > 0:
        val_data.loc[val_data[time_col] > censor_at, event_col] = 0
        val_data.loc[val_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    val_data.loc[val_data[event_col] > 1, event_col] = 0

     ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        pass
    
    val_data_temp = val_data[feats_list_temp]
    # print(val_data_temp)
    partial_hazard_test = cph_mva.predict_partial_hazard(val_data_temp)
    # c-index on the validation set
    try:
        vcindex = concordance_index(val_data_temp[time_col], -partial_hazard_test, val_data_temp[event_col])
    except Exception as e:
        print(f'Evaluation failed: {e}')
        return -1

    # Use mean value in the discovery set as the cut-off value and divide subjects int the validation set into two groups
    upper = partial_hazard_test >= cutoff_value
    T_upper_test = val_data_temp[time_col][upper]
    E_upper_test = val_data_temp[event_col][upper]
    lower = partial_hazard_test < cutoff_value
    T_lower_test = val_data_temp[time_col][lower]
    E_lower_test = val_data_temp[event_col][lower]

    # Log-rank test: if there is any significant difference between the groups being compared
    results = logrank_test(T_lower_test, T_upper_test, E_lower_test, E_upper_test)
    
    val_pvalue = results.p_value
    val_cindex = vcindex
    val_hr = cph_mva.hazard_ratios_

    if plotit:
        val_fig = plot_km(val_data_temp[time_col],val_data_temp[event_col],partial_hazard_test,cutoff_value)
    
    if plotit:
        return val_hr, val_cindex, val_pvalue, train_fig, val_fig
    return val_hr, val_cindex, val_pvalue

    
    
##fitting a CoxPH model on the discovery set and validating on the validation set
def CoxPH_train_val(path_to_train_set, path_to_val_set, path_to_save_results, feats_list, time_col, event_col, subset, censor_at, plotit, cutoff_mode, cutoff_point, hr_scores, split):
    train_data = path_to_train_set
    #train_data = train_data.dropna()
    train_data = train_data.fillna(0)

    ##apply censoring. Commenting these out as we can use the uncensored data for training but for validation will need to censor
    #train_data.loc[train_data[time_col] > censor_at, event_col] = 0
    #train_data.loc[train_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    train_data.loc[train_data[event_col] > 1, event_col] = 0

    ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)

    ##these lines can be commented out (set 2 == 3 in the if) to train the model using lymph node 0-3 so that more event are included during training even though the validation might be done on LN- only.
    if subset == 'Endocrine_LN0' and 2 == 2:
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)
        train_data.drop(train_data[train_data['Lymph Node status'] == 1].index, inplace=True)
    
    feats_list_temp = []
    
    if isinstance(feats_list, str):
        feats_list_temp = [feats_list, time_col, 'event']
    else:
        for l in feats_list:
            feats_list_temp.append(l)

        feats_list_temp.append(time_col)
        feats_list_temp.append(event_col)

    train_data_temp = train_data[feats_list_temp]

    trn_ids = train_data['Case ID']
    #train_data = train_data.drop(columns=['wsi_id', time_col, event_col])
    #train_data_infer = train_data_infer.drop(columns=[event_col, time_col])

    # print('Train data: ', train_data_temp.shape)

    estim_method = 'breslow' ## 'breslow', 'spline', 'piecewise'
    l1_r = 0.5 ## specify what ratio to assign to a L1 vs L2 penalty.
    penalty = 0.001 #0.001 #0.001 was used for the submission 2, 3, and 4 in the laptop R submission folder. D:/warwick/tasks/Nottingham_Exemplar/ML/cellular_analysis/prelim_val_sets_23_28_n_challenging
    
    try:
        cph_mva = CoxPHFitter(baseline_estimation_method=estim_method, l1_ratio=l1_r, penalizer=penalty).fit(train_data_temp, duration_col=time_col, event_col=event_col)
    except:
        print('Failed in fitting CoxPH for train_data_infer')
        return 999, -1, -1, -1, -1, 999, -1, -1, -1, -1, cutoff_point, hr_scores

    #cph_mva.print_summary() ##print the summary of the fit #####@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    train_data_infer = train_data_temp.drop(columns=[event_col, time_col])
    partial_hazard_train = cph_mva.predict_partial_hazard(train_data_infer)
    
    # Use mean value in the discovery set as the cut-off value and divide subjects into two groups
    if cutoff_point == -1:
        if cutoff_mode == 'mean':
            cutoff_value = partial_hazard_train.mean()
        elif cutoff_mode == 'median':
            cutoff_value = partial_hazard_train.median()
        elif cutoff_mode == 'quantile':
            cutoff_value = partial_hazard_train.quantile(0.60)
        else:
            cutoff_value = cutoff_mode
    else:
        cutoff_value = cutoff_point
    
    ##save the hazard score as a new feature for submission to evaluation
    partial_hazard_temp = pd.DataFrame()
    partial_hazard_temp['TGCI'] = partial_hazard_train
    trn_ids = trn_ids.str.split('_', expand=True)[0] 
    partial_hazard_temp['WSI_ID'] = train_data['Case ID'] #trn_ids 

    ##add the extra features for HR forest plots in R  ####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    feat_to_add = ['per_g1','per_g2','per_g3','tumor_per_all_regionpatch','Grade','M','NPI','P','T','LVI','Histological Tumour Type','Lymph Node status','Number of Positive LNs','Invasive Tumour Size (cm)','Associated DCIS','Associated LCIS','Multifocality','Age at Diagnosis','Menopausal status','Survival Status','Breast cancer specific survival/ month','Distant Metastasis','TTDM/ month','Chemotherapy','Endocrine Therapy']
    
    for f in feat_to_add:
        partial_hazard_temp[f] = train_data[f]   

    partial_hazard_temp[time_col] = train_data[time_col]
    partial_hazard_temp[event_col] = train_data[event_col]
    
    partial_hazard_temp.to_csv(path_to_save_results + 'TGCI_discov.csv')
    partial_hazard_train_temp = partial_hazard_temp

    upper = partial_hazard_train >= cutoff_value
    T_upper_train = train_data_temp[time_col][upper]
    E_upper_train = train_data_temp[event_col][upper]
    lower = partial_hazard_train < cutoff_value
    T_lower_train = train_data_temp[time_col][lower]
    E_lower_train = train_data_temp[event_col][lower]
    # Initializing the KaplanMeierModel for each group
    km_upper = KaplanMeierFitter()
    km_lower = KaplanMeierFitter()
    
    # Log-rank test: if there is any significant difference between the groups being compared
    from lifelines.statistics import logrank_test
    results = logrank_test(T_lower_train, T_upper_train, E_lower_train, E_upper_train)
    train_data_temp['hazard'] = partial_hazard_train
    dataset_train = train_data_temp[['hazard', time_col, event_col]]
    try:
        cph = CoxPHFitter(baseline_estimation_method=estim_method, l1_ratio=l1_r, penalizer=penalty).fit(dataset_train, time_col, event_col)
    except:
        print('Failed in fitting CoxPH for train_data_temp')
        return 999, -1, -1, -1, -1, 999, -1, -1, -1, -1, cutoff_value, hr_scores
    
    tcindex = cph.concordance_index_
    thzratio = cph.hazard_ratios_
    thzratio = thzratio.hazard
    
    try:
        thz_ci_low = math.exp(cph.confidence_intervals_.values[0][0])
    except:
        thz_ci_low = float('inf')
    
    try:
        thz_ci_high = math.exp(cph.confidence_intervals_.values[0][1])
    except:
        thz_ci_high = float('inf')

    train_pvalue = results.p_value
    train_cindex = tcindex
    
    if plotit:
        font_size = 18
        fig_size = 10
        fig = plt.figure(figsize=(fig_size, fig_size-2)) ##adjust according to font size
        ax = fig.add_subplot(111)

        ax.set_xlabel('', fontsize=font_size)
        ax.set_ylabel('', fontsize=font_size)
            
        if train_pvalue < 0.0001:
            train_pvalue_txt = 'p < 0.0001'
        else:
            train_pvalue_txt = 'p = ' + str(np.round(train_pvalue, 4))

        from matplotlib.offsetbox import AnchoredText
        ax.add_artist(AnchoredText(train_pvalue_txt, loc=1, frameon=False, prop=dict(size=font_size)))
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        if event_col == 'Distant Metastasis':
            event_type = 'DMFS'
        elif event_col == 'Survival Status':
            event_type = 'BCSS'

        ax = km_upper.fit(T_upper_train, event_observed=E_upper_train, label='high').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='r', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        ax = km_lower.fit(T_lower_train, event_observed=E_lower_train, label='low').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='b', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(km_upper, km_lower, ax=ax, fig=fig, fontsize=int(font_size*1) )
        ax.get_legend().remove()
        plt.subplots_adjust(bottom=0.4)
        plt.subplots_adjust(left=0.2)
        #plt.show()
        
        plt.savefig(path_to_save_results + event_type + '_train_' + str(split) + '.pdf', format='pdf', dpi=600)
            
    val_data = path_to_val_set #pd.read_csv(path_to_val_set)
    #val_data = val_data.dropna()
    val_data = val_data.fillna(0)

    ##apply censoring. 
    if censor_at > 0:
        val_data.loc[val_data[time_col] > censor_at, event_col] = 0
        val_data.loc[val_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    val_data.loc[val_data[event_col] > 1, event_col] = 0

     ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        val_data.drop(val_data[val_data['Endocrine Therapy'] != 1].index, inplace=True)
        val_data.drop(val_data[val_data['Chemotherapy'] == 1].index, inplace=True)

    if subset == 'Endocrine_LN0':
        val_data.drop(val_data[val_data['Endocrine Therapy'] != 1].index, inplace=True)
        val_data.drop(val_data[val_data['Chemotherapy'] == 1].index, inplace=True)
        val_data.drop(val_data[val_data['Lymph Node status'] == 1].index, inplace=True)
    
    val_data_temp = val_data[feats_list_temp]
    val_ids = val_data['Case ID']
   
    partial_hazard_test = cph_mva.predict_partial_hazard(val_data_temp)

    partial_hazard_temp = pd.DataFrame()
    partial_hazard_temp['TGCI'] = partial_hazard_test
    # val_ids = val_ids.str.split('_', expand=True)[0]
    partial_hazard_temp['WSI_ID'] = val_ids

    partial_hazard_temp.to_csv(path_to_save_results + 'TGCI' + str(split) + '.csv') ## to save results for preliminary validation sets ####@@@@@@@@@@@@@@@@@@@@@@@@@@@
        
    ##add the extra features for HR comparison in R  ####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    feat_to_add = ['per_g1','per_g2','per_g3','tumor_per_all_regionpatch','Grade','M','NPI','P','T','LVI','Histological Tumour Type','Lymph Node status','Number of Positive LNs','Invasive Tumour Size (cm)','Associated DCIS','Associated LCIS','Multifocality','PgR STATUS','Age at Diagnosis','Menopausal status','Survival Status','Breast cancer specific survival/ month','Distant Metastasis','TTDM/ month','Chemotherapy','Endocrine Therapy','Local Recurrence','TTLR/ month','Overall Recurrence events','Disease Free-Survival/ month']

    for f in feat_to_add:
        partial_hazard_temp[f] = val_data[f]       

    partial_hazard_temp.to_csv(path_to_save_results + 'TGCI_valid.csv')

    # Use mean value in the discovery set as the cut-off value and divide subjects int the validation set into two groups
    upper = partial_hazard_test >= cutoff_value
    T_upper_test = val_data_temp[time_col][upper]
    E_upper_test = val_data_temp[event_col][upper]
    lower = partial_hazard_test < cutoff_value
    T_lower_test = val_data_temp[time_col][lower]
    E_lower_test = val_data_temp[event_col][lower]
    # Initializing the KaplanMeierModel for each group
    km_upper = KaplanMeierFitter()
    km_lower = KaplanMeierFitter()

    # Log-rank test: if there is any significant difference between the groups being compared
    results = logrank_test(T_lower_test, T_upper_test, E_lower_test, E_upper_test)

    #print("p-value on the validation set %s; log-rank %s" % (results.p_value, np.round(results.test_statistic, 6)))
    pd.options.mode.chained_assignment = None
    val_data_temp['hazard'] = partial_hazard_test
    dataset_test = val_data_temp[['hazard', time_col, event_col]]
    try:
        cph = CoxPHFitter(baseline_estimation_method=estim_method, l1_ratio=l1_r, penalizer=penalty).fit(dataset_test, time_col, event_col)  #####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ commented this out to see why we fit this again as once on the train set it is done above.
    except:
        print('Failed in fitting CoxPH for train_data_temp')
        return 999, -1, -1, -1, -1, 999, -1, -1, -1, -1, cutoff_value, hr_scores
    
    #vcindex = concordance_index(dataset_test['time'], -cph.predict_partial_hazard(dataset_test), dataset_test['event'])
    vcindex = cph.concordance_index_
    vhzratio = cph.hazard_ratios_
    vhzratio = vhzratio.hazard
    try:
        vhz_ci_low = math.exp(cph.confidence_intervals_.values[0][0])
    except:
        vhz_ci_low = float('inf')
    
    try:
        vhz_ci_high = math.exp(cph.confidence_intervals_.values[0][1])
    except:
        vhz_ci_high = float('inf')

    val_pvalue = results.p_value
    val_cindex = vcindex

    if plotit:
        font_size = 18
        fig_size = 10
        fig = plt.figure(figsize=(fig_size, fig_size-2)) ##adjust according to font size
        ax = fig.add_subplot(111)

        ax.set_xlabel('', fontsize=font_size)
        ax.set_ylabel('', fontsize=font_size)
            
        if val_pvalue < 0.0001:
            val_pvalue_txt = 'p < 0.0001'
        else:
            val_pvalue_txt = 'p = ' + str(np.round(val_pvalue, 4))

        from matplotlib.offsetbox import AnchoredText
        ax.add_artist(AnchoredText(val_pvalue_txt, loc=1, frameon=False, prop=dict(size=font_size)))
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        if event_col == 'Distant Metastasis':
            event_type = 'DMFS'
        elif event_col == '':
            event_type = 'BCSS'

        ax = km_upper.fit(T_upper_test, event_observed=E_upper_test, label='high').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='r', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        ax = km_lower.fit(T_lower_test, event_observed=E_lower_test, label='low').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='b', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(km_upper, km_lower, ax=ax, fig=fig, fontsize=int(font_size*1) )
        ax.get_legend().remove()
        plt.subplots_adjust(bottom=0.4)
        plt.subplots_adjust(left=0.2)
        #plt.show()
        
        plt.savefig(path_to_save_results + event_type + '_val_' + str(split) + '.pdf', format='pdf', dpi=600)

    # print('Valid data: ', val_data_temp.shape)
           
    return train_pvalue, train_cindex, thzratio, thz_ci_low, thz_ci_high, val_pvalue, val_cindex, vhzratio, vhz_ci_low, vhz_ci_high, cutoff_value, hr_scores

##fitting a CoxPH model on the discovery set and generating risk score on the test set.
def CoxPH_train_infer(path_to_train_set, path_to_val_set, path_to_save_results, feats_list, time_col, event_col, subset, censor_at, plotit, cutoff_mode, cutoff_point, hr_scores, split):
    train_data = path_to_train_set #pd.read_csv(path_to_train_set)
    #train_data = train_data.dropna()
    train_data = train_data.fillna(0)

    ##apply censoring. Commenting these out as we can use the uncensored data for training but for validation will need to censor
    #train_data.loc[train_data[time_col] > censor_at, event_col] = 0
    #train_data.loc[train_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    train_data.loc[train_data[event_col] > 1, event_col] = 0

    ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)

    ##these lines can be commented out (set 2 == 3 in the if) to train the model using lymph node 0-3 so that more event are included during training even though the validation might be done on LN- only.
    if subset == 'Endocrine_LN0' and 2 == 2:
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)
        train_data.drop(train_data[train_data['Lymph Node status'] == 1].index, inplace=True)
    
    feats_list_temp = []
    
    if isinstance(feats_list, str):
        feats_list_temp = [feats_list, time_col, 'event']
    else:
        for l in feats_list:
            feats_list_temp.append(l)

        feats_list_temp.append(time_col)
        feats_list_temp.append(event_col)

    train_data_temp = train_data[feats_list_temp]

    trn_ids = train_data['Case ID']
    #train_data = train_data.drop(columns=['wsi_id', time_col, event_col])
    #train_data_infer = train_data_infer.drop(columns=[event_col, time_col])

    print('Train data: ', train_data_temp.shape)

    estim_method = 'breslow' ## 'breslow', 'spline', 'piecewise'
    l1_r = 0.3 ## specify what ratio to assign to a L1 vs L2 penalty.
    penalty = 0.001 #0.001 #0.001 was used for the submission 2, 3, and 4 in the laptop R submission folder. D:/warwick/tasks/Nottingham_Exemplar/ML/cellular_analysis/prelim_val_sets_23_28_n_challenging
    
    try:
        cph_mva = CoxPHFitter(baseline_estimation_method=estim_method, l1_ratio=l1_r, penalizer=penalty).fit(train_data_temp, duration_col=time_col, event_col=event_col)
    except:
        print('Failed in fitting CoxPH for train_data_infer')
        return 999, -1, -1, -1, -1, 999, -1, -1, -1, -1, cutoff_point, hr_scores

    #cph_mva.print_summary() ##print the summary of the fit #####@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    train_data_infer = train_data_temp.drop(columns=[event_col, time_col])
    partial_hazard_train = cph_mva.predict_partial_hazard(train_data_infer)
    
    # Use mean value in the discovery set as the cut-off value and divide subjects into two groups
    if cutoff_point == -1:
        if cutoff_mode == 'mean':
            cutoff_value = partial_hazard_train.mean()
        elif cutoff_mode == 'median':
            cutoff_value = partial_hazard_train.median()
        elif cutoff_mode == 'quantile':
            cutoff_value = partial_hazard_train.quantile(0.60)
        else:
            cutoff_value = cutoff_mode
    else:
        cutoff_value = cutoff_point
    
    ##save the hazard score as a new feature for submission to evaluation
    partial_hazard_temp = pd.DataFrame()
    partial_hazard_temp['TGCI'] = partial_hazard_train
    trn_ids = trn_ids.str.split('_', expand=True)[0] 
    partial_hazard_temp['WSI_ID'] = train_data['Case ID'] #trn_ids 

    ##add the extra features for HR forest plots in R  ####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    feat_to_add = ['per_g1','per_g2','per_g3','tumor_per_all_regionpatch','Grade','M','NPI','P','T','LVI','Histological Tumour Type','Lymph Node status','Number of Positive LNs','Invasive Tumour Size (cm)','Associated DCIS','Associated LCIS','Multifocality','Age at Diagnosis','Menopausal status','Survival Status','Breast cancer specific survival/ month','Distant Metastasis','TTDM/ month','Chemotherapy','Endocrine Therapy']
    
    for f in feat_to_add:
        partial_hazard_temp[f] = train_data[f]   

    partial_hazard_temp[time_col] = train_data[time_col]
    partial_hazard_temp[event_col] = train_data[event_col]
    
    partial_hazard_temp.to_csv(path_to_save_results + 'TGCI_discov.csv')
    partial_hazard_train_temp = partial_hazard_temp

    upper = partial_hazard_train >= cutoff_value
    T_upper_train = train_data_temp[time_col][upper]
    E_upper_train = train_data_temp[event_col][upper]
    lower = partial_hazard_train < cutoff_value
    T_lower_train = train_data_temp[time_col][lower]
    E_lower_train = train_data_temp[event_col][lower]
    # Initializing the KaplanMeierModel for each group
    km_upper = KaplanMeierFitter()
    km_lower = KaplanMeierFitter()
    
    # Log-rank test: if there is any significant difference between the groups being compared
    from lifelines.statistics import logrank_test
    results = logrank_test(T_lower_train, T_upper_train, E_lower_train, E_upper_train)
    train_data_temp['hazard'] = partial_hazard_train
    dataset_train = train_data_temp[['hazard', time_col, event_col]]
    try:
        cph = CoxPHFitter(baseline_estimation_method=estim_method, l1_ratio=l1_r, penalizer=penalty).fit(dataset_train, time_col, event_col)
    except:
        print('Failed in fitting CoxPH for train_data_temp')
        return 999, -1, -1, -1, -1, 999, -1, -1, -1, -1, cutoff_value, hr_scores
    
    tcindex = cph.concordance_index_
    thzratio = cph.hazard_ratios_
    thzratio = thzratio.hazard
    
    try:
        thz_ci_low = math.exp(cph.confidence_intervals_.values[0][0])
    except:
        thz_ci_low = float('inf')
    
    try:
        thz_ci_high = math.exp(cph.confidence_intervals_.values[0][1])
    except:
        thz_ci_high = float('inf')

    train_pvalue = results.p_value
    train_cindex = tcindex
    
    if plotit:
        font_size = 18
        fig_size = 10
        fig = plt.figure(figsize=(fig_size, fig_size-2)) ##adjust according to font size
        ax = fig.add_subplot(111)

        ax.set_xlabel('', fontsize=font_size)
        ax.set_ylabel('', fontsize=font_size)
            
        if train_pvalue < 0.0001:
            train_pvalue_txt = 'p < 0.0001'
        else:
            train_pvalue_txt = 'p = ' + str(np.round(train_pvalue, 4))

        from matplotlib.offsetbox import AnchoredText
        ax.add_artist(AnchoredText(train_pvalue_txt, loc=1, frameon=False, prop=dict(size=font_size)))
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        if event_col == 'Distant Metastasis':
            event_type = 'DMFS'
        elif event_col == 'Survival Status':
            event_type = 'BCSS'

        ax = km_upper.fit(T_upper_train, event_observed=E_upper_train, label='high').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='r', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        ax = km_lower.fit(T_lower_train, event_observed=E_lower_train, label='low').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='b', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(km_upper, km_lower, ax=ax, fig=fig, fontsize=int(font_size*1) )
        ax.get_legend().remove()
        plt.subplots_adjust(bottom=0.4)
        plt.subplots_adjust(left=0.2)
        #plt.show()
        
        plt.savefig(path_to_save_results + event_type + '_train_' + str(split) + '.pdf', format='pdf', dpi=600)
            
    val_data = path_to_val_set #pd.read_csv(path_to_val_set)
    #val_data = val_data.dropna()
    val_data = val_data.fillna(0)

    feats_list_temp.remove(time_col)
    feats_list_temp.remove(event_col)
    
    val_data_temp = val_data[feats_list_temp]
    val_ids = val_data['Case ID']
   
    partial_hazard_test = cph_mva.predict_partial_hazard(val_data_temp)

    partial_hazard_temp = pd.DataFrame()
    partial_hazard_temp['TGCI'] = partial_hazard_test
    val_ids = val_ids.str.split('_', expand=True)[0]
    partial_hazard_temp['WSI_ID'] = val_data['Case ID'] #val_ids 

    partial_hazard_temp.to_csv(path_to_save_results + 'TGCI_test.csv')

    print('Valid data: ', val_data_temp.shape)
           
    return train_pvalue, train_cindex, thzratio, thz_ci_low, thz_ci_high, cutoff_value, hr_scores

##usng clinical features for stratification of discovery and validation sets
def clinic_eval(path_to_train_set, path_to_val_set, path_to_save_results, feats_list, time_col, event_col, subset, censor_at, plotit, cutoff_mode, cutoff_point, hr_scores, split):
    train_data = path_to_train_set
    #train_data = train_data.dropna()
    train_data = train_data.fillna(0)

    ##apply censoring. Commenting these out as we can use the uncensored data for training but for validation will need to censor
    #train_data.loc[train_data[time_col] > censor_at, event_col] = 0
    #train_data.loc[train_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    train_data.loc[train_data[event_col] > 1, event_col] = 0

    ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)

    ##these lines can be commented out (set 2 == 3 in the if) to train the model using lymph node 0-3 so that more event are included during training even though the validation might be done on LN- only.
    if subset == 'Endocrine_LN0' and 2 == 2:
        train_data.drop(train_data[train_data['Endocrine Therapy'] != 1].index, inplace=True)
        train_data.drop(train_data[train_data['Chemotherapy'] == 1].index, inplace=True)
        train_data.drop(train_data[train_data['Lymph Node status'] == 1].index, inplace=True)
    
    feats_list_temp = []
    
    if isinstance(feats_list, str):
        feats_list_temp = [feats_list, time_col, 'event']
    else:
        for l in feats_list:
            feats_list_temp.append(l)

        feats_list_temp.append(time_col)
        feats_list_temp.append(event_col)

    train_data_temp = train_data[feats_list_temp]

    trn_ids = train_data['Case ID']
    #train_data = train_data.drop(columns=['wsi_id', time_col, event_col])
    #train_data_infer = train_data_infer.drop(columns=[event_col, time_col])

    print('Train data: ', train_data_temp.shape)

    #cph_mva.print_summary() ##print the summary of the fit #####@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    #train_data_infer = train_data_temp.drop(columns=[event_col, time_col])
    
    # Use mean value in the discovery set as the cut-off value and divide subjects into two groups
    if cutoff_point == -1:
        if cutoff_mode == 'mean':
            cutoff_value = train_data.mean()
        elif cutoff_mode == 'median':
            cutoff_value = train_data.median()
        elif cutoff_mode == 'quantile':
            cutoff_value = train_data.quantile(0.60)
        else:
            cutoff_value = cutoff_mode
    else:
        cutoff_value = cutoff_point
    
    upper = train_data[feats_list[0]] >= cutoff_value
    T_upper_train = train_data_temp[time_col][upper]
    E_upper_train = train_data_temp[event_col][upper]
    lower = train_data[feats_list[0]] < cutoff_value
    T_lower_train = train_data_temp[time_col][lower]
    E_lower_train = train_data_temp[event_col][lower]
    # Initializing the KaplanMeierModel for each group
    km_upper = KaplanMeierFitter()
    km_lower = KaplanMeierFitter()
    
    # Log-rank test: if there is any significant difference between the groups being compared
    from lifelines.statistics import logrank_test
    results = logrank_test(T_lower_train, T_upper_train, E_lower_train, E_upper_train)

    train_data['event'] = train_data[event_col]
    train_data['time'] = train_data[time_col]
    train_cindex = calc_cindex(train_data, feats_list[0])[0]
    train_pvalue = results.p_value
        
    if plotit:
        font_size = 18
        fig_size = 10
        fig = plt.figure(figsize=(fig_size, fig_size-2)) ##adjust according to font size
        ax = fig.add_subplot(111)

        ax.set_xlabel('', fontsize=font_size)
        ax.set_ylabel('', fontsize=font_size)
            
        if train_pvalue < 0.0001:
            train_pvalue_txt = 'p < 0.0001'
        else:
            train_pvalue_txt = 'p = ' + str(np.round(train_pvalue, 4))

        from matplotlib.offsetbox import AnchoredText
        ax.add_artist(AnchoredText(train_pvalue_txt, loc=1, frameon=False, prop=dict(size=font_size)))
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        if event_col == 'Distant Metastasis':
            event_type = 'DMFS'
        elif event_col == 'Survival Status':
            event_type = 'BCSS'

        ax = km_upper.fit(T_upper_train, event_observed=E_upper_train, label='high').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='r', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        ax = km_lower.fit(T_lower_train, event_observed=E_lower_train, label='low').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='b', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(km_upper, km_lower, ax=ax, fig=fig, fontsize=int(font_size*1) )
        ax.get_legend().remove()
        plt.subplots_adjust(bottom=0.4)
        plt.subplots_adjust(left=0.2)
        #plt.show()
        
        plt.savefig(path_to_save_results + event_type + '_train_' + str(split) + '.pdf', format='pdf', dpi=600)
            
    val_data = path_to_val_set #pd.read_csv(path_to_val_set)
    #val_data = val_data.dropna()
    val_data = val_data.fillna(0)

    ##apply censoring. 
    if censor_at > 0:
        val_data.loc[val_data[time_col] > censor_at, event_col] = 0
        val_data.loc[val_data[time_col] > censor_at, time_col] = censor_at

    ##convert all events other than 1 to 0
    val_data.loc[val_data[event_col] > 1, event_col] = 0

     ##apply lymph node filtering i.e consider lymph node negative, or consider lymph node 0-3
    if subset == 'Endocrine':
        val_data.drop(val_data[val_data['Endocrine Therapy'] != 1].index, inplace=True)
        val_data.drop(val_data[val_data['Chemotherapy'] == 1].index, inplace=True)

    if subset == 'Endocrine_LN0':
        val_data.drop(val_data[val_data['Endocrine Therapy'] != 1].index, inplace=True)
        val_data.drop(val_data[val_data['Chemotherapy'] == 1].index, inplace=True)
        val_data.drop(val_data[val_data['Lymph Node status'] == 1].index, inplace=True)
    
    val_data_temp = val_data[feats_list_temp]
    val_ids = val_data['Case ID']
   
    # Use mean value in the discovery set as the cut-off value and divide subjects int the validation set into two groups
    upper = val_data_temp[feats_list[0]] >= cutoff_value
    T_upper_test = val_data_temp[time_col][upper]
    E_upper_test = val_data_temp[event_col][upper]
    lower = val_data_temp[feats_list[0]] < cutoff_value
    T_lower_test = val_data_temp[time_col][upper]
    E_lower_test = val_data_temp[event_col][upper]
    # Initializing the KaplanMeierModel for each group
    km_upper = KaplanMeierFitter()
    km_lower = KaplanMeierFitter()

    # Log-rank test: if there is any significant difference between the groups being compared
    results = logrank_test(T_lower_test, T_upper_test, E_lower_test, E_upper_test)

    #print("p-value on the validation set %s; log-rank %s" % (results.p_value, np.round(results.test_statistic, 6)))
    pd.options.mode.chained_assignment = None
    
    val_data_temp['event'] = val_data_temp[event_col]
    val_data_temp['time'] = val_data_temp[time_col]
    val_cindex = calc_cindex(val_data_temp, feats_list[0])[0]
    val_pvalue = results.p_value

    if plotit:
        font_size = 18
        fig_size = 10
        fig = plt.figure(figsize=(fig_size, fig_size-2)) ##adjust according to font size
        ax = fig.add_subplot(111)

        ax.set_xlabel('', fontsize=font_size)
        ax.set_ylabel('', fontsize=font_size)
            
        if val_pvalue < 0.0001:
            val_pvalue_txt = 'p < 0.0001'
        else:
            val_pvalue_txt = 'p = ' + str(np.round(val_pvalue, 4))

        from matplotlib.offsetbox import AnchoredText
        ax.add_artist(AnchoredText(val_pvalue_txt, loc=1, frameon=False, prop=dict(size=font_size)))
        ax.tick_params(axis='x', labelsize=font_size)
        ax.tick_params(axis='y', labelsize=font_size)

        if event_col == 'Distant Metastasis':
            event_type = 'DMFS'
        elif event_col == '':
            event_type = 'BCSS'

        ax = km_upper.fit(T_upper_test, event_observed=E_upper_test, label='high').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='r', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        ax = km_lower.fit(T_lower_test, event_observed=E_lower_test, label='low').plot_survival_function(ax=ax, show_censors=True, censor_styles={'ms': 5}, color='b', ci_show=False, xlabel='Months', ylabel= event_type + ' Probability')
        from lifelines.plotting import add_at_risk_counts
        add_at_risk_counts(km_upper, km_lower, ax=ax, fig=fig, fontsize=int(font_size*1) )
        ax.get_legend().remove()
        plt.subplots_adjust(bottom=0.4)
        plt.subplots_adjust(left=0.2)
        #plt.show()
        
        plt.savefig(path_to_save_results + event_type + '_val_' + str(split) + '.pdf', format='pdf', dpi=600)

    print('Valid data: ', val_data_temp.shape)
    thzratio = thz_ci_low = thz_ci_high = vhzratio = vhz_ci_low = vhz_ci_high = 0
           
    return train_pvalue, train_cindex, thzratio, thz_ci_low, thz_ci_high, val_pvalue, val_cindex, vhzratio, vhz_ci_low, vhz_ci_high, cutoff_value, hr_scores


##combine features from multiple csv files into on csv file
def combine_csvs(csv_list, path_to_clinical_csv, path_to_save_combined_csv, skip_list):
    feats_all = pd.DataFrame()
    trn_df = pd.read_csv(path_to_clinical_csv)
    trn_df['wsi_id'] = trn_df['Case ID'].str.split('_', expand=True)[0]
    trn_ids = trn_df['wsi_id'] #trn_df['Case ID'].str.split('_', expand=True)[0]
    id_list = {}
    for f in csv_list:
        frm = pd.read_csv(f)
        ids_temp = frm['wsi_id'].str.split('_', expand=True)[0]
        ids = []
        for i in ids_temp:
            if i.split('_')[0] in skip_list:
                continue
            ids.append(i.split('_')[0])  ### comment this when processing oncodx set
            
        id_list[f] = ids

    filtered_ids = []
    
    for ti in trn_ids:
        cond = True
        ##check if this case ID exists in all the csv files
        for f in csv_list:
            ids = id_list[f]
            if ti not in ids:
                cond = False
                break
        if cond == False:
            continue
        else:
            filtered_ids.append(ti)
    
    filtered_ids_df = pd.DataFrame(filtered_ids, columns=['wsi_id'])
    print(trn_df.shape)
    
    trn_df = trn_df.merge(filtered_ids_df, how='inner', on='wsi_id')

    print(trn_df.shape)

    for f in csv_list:
        temp_df = pd.read_csv(f)
        frm_id = temp_df['wsi_id'].str.split('_', expand=True)[0]
        temp_df['wsi_id'] = frm_id

        trn_df = trn_df.merge(temp_df, how='inner', on='wsi_id')
        print(trn_df.shape)

    trn_df.to_csv(path_to_save_combined_csv, index=False)
    temp_file = pd.read_csv(path_to_save_combined_csv)
    temp_file.drop(temp_file.filter(regex="Unname"),axis=1, inplace=True)
    temp_file.to_csv(path_to_save_combined_csv, index=False)

##combined features from multiple WSIs per case (for examples, cases in Challenging set have multiple WSIs and some of these are also part of other sets)
## In this case we are taking the max of features across the multiple WSIs.
def combine_feats_multipe_WSI(path_csv_file):
    ##combined the features from multiple WSI per case. This is the case with challenging sets of BRACE. No other set has any case with multiple WSIs.
    
    mult_csv = pd.read_csv(path_csv_file)
    combined_csv = pd.DataFrame()

    w_ids = mult_csv['wsi_id'].str.split('_', expand=True)
    w_ids = w_ids[0].drop_duplicates().tolist()
    #print(mult_csv.columns)

    for w in w_ids:
        mult_wsi_rows = mult_csv[mult_csv['wsi_id'].str.contains(w)].max()
        temp_feat = {}
        temp_feat['wsi_id'] = w
        for j in mult_csv.columns:
            if j == 'wsi_id':
                continue
            temp_feat[j] = mult_wsi_rows[j]

        combined_csv = combined_csv.append(temp_feat, ignore_index=True)

    combined_csv.to_csv(path_csv_file.split('.csv')[0] + '_combined.csv', index=False)

    print('Done meta aggregation')

##calculate c-index for clinical feature when no model is fitted but just the raw features used for stratification
def calc_cindex(data, feat):
    ##for c-index calculation
    r = data['event']
    d = data['time']
    data_dtype = [('Status', np.bool), ('Survival_in_days', np.int32)]
    data_y = np.array([], dtype=data_dtype)
            
    lbls = np.array([])
    for i, j in zip(r, d):
        if i == 1:
            data_y = np.append(data_y, np.array([(True, j)], dtype=data_dtype))
            lbls = np.append(lbls, 1)
        elif i == 0:
            data_y = np.append(data_y, np.array([(False, j)], dtype=data_dtype))
            lbls = np.append(lbls, 1)
                
    return concordance_index_censored(data_y["Status"], data_y["Survival_in_days"], data[feat])