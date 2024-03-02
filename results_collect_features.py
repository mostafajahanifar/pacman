import pandas as pd
from tqdm import tqdm

censor_at = 60
cv_experiment = f'CV_results_5years_median'

feats_list = pd.read_csv('noncorrolated_feature_list.csv', header=None)[0].to_list()
feat_appearance_dict = {'cancer_type': [], 'event_type': [], 'success': []}
print(feats_list)
for feat in feats_list:
    feat_appearance_dict[feat] = []

event_types = ['DFI', 'PFI', 'OS', 'DSS']
for event_type in event_types:
    cancer_types = [["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"], ["GBM"], ["HNSC"], ["KICH"], ["KIRC"], ["KIRP"], ["LGG"], ["LIHC"], ["LUAD"], ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"]]

    csv_files = {}
    sig_p_values = {}
    for cancer_type in cancer_types:
        csv_path = f"{cv_experiment}/CV_{cancer_type}/bootstrap_results_{cancer_type}_{event_type}_censor{censor_at}.csv"
        csv_files[''.join(cancer_type)] = csv_path
        feat_appearance_dict['cancer_type'].append(''.join(cancer_type).upper())
        feat_appearance_dict['event_type'].append(event_type)
        try:
            exp_df = pd.read_csv(csv_path)
        except:
            feat_appearance_dict['success'].append(0)
            for feat in feats_list:
                feat_appearance_dict[feat].append(0)
            continue

        if len(exp_df) > 500:
            feat_appearance_dict['success'].append(1)
        else:
            feat_appearance_dict['success'].append(0)
        this_set = set(exp_df.columns) - set(['c_index', 'p_value'])
        for feat in feats_list:
            feat_appearance_dict[feat].append(1 if feat in this_set else 0)

feat_appearance_df = pd.DataFrame(feat_appearance_dict)

feat_appearance_df.to_excel(f'{cv_experiment}/features_appearance.xlsx')

fad = feat_appearance_df[feat_appearance_df['success']==1]

summ_dict = {'feature': []}
for event_type in event_types:
    summ_dict[event_type] = []

for feat in feats_list:
    summ_dict['feature'].append(feat)
    for event_type in event_types:
        df = fad[fad['event_type']==event_type]
        summ_dict[event_type].append(df[feat].sum())
        
summ_df = pd.DataFrame(summ_dict)
summ_df['Overall'] = summ_df[event_types].sum(axis=1)
summ_df = summ_df.sort_values(by='Overall', ascending=False)
print(summ_df)
summ_df.to_excel(f'{cv_experiment}/features_appearance_summary.xlsx')
