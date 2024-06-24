import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from pathlib import Path
import numpy as np
from utils import get_colors_dict


censor_at = -1
cv_experiment = f'CV_results_NoCensor_median'

results_dict = {'censoring': [],
                'event_type': [],
                'cancer_type': [],
                'cindex_mean': [],
                'cindex_std': [],
                'pvalue':[],
                'success':[],}

event_types = ['DFI', 'PFI', 'OS', 'DSS']
for event_type in event_types:
    cancer_types = [["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"], ["GBM"], ["HNSC"], ["KICH"], ["KIRC"], ["KIRP"], ["LGG"], ["LIHC"], ["LUAD"], ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"]]

    csv_files = {}
    sig_p_values = {}
    for cancer_type in cancer_types:
        results_dict['censoring'].append(censor_at)
        results_dict['event_type'].append(event_type)
        results_dict['cancer_type'].append(''.join(cancer_type).upper())
        

        # find the p-value related to this experiment
        directory = f"{cv_experiment}/CV_{cancer_type}/" # cv_results_{cancer_type}_{event_type}_censor{censor_at}_*.csv"
        file = f"bootstrap_results_{cancer_type}_{event_type}_censor{censor_at}.csv"
        try:
            csv_df = pd.read_csv(directory+file)
        except:
            results_dict['success'].append(0)
            results_dict['pvalue'].append(1)
            results_dict['cindex_mean'].append(0)
            results_dict['cindex_std'].append(0)
            continue

        results_dict['pvalue'].append(csv_df['p_value'].median())
        results_dict['cindex_mean'].append(csv_df['c_index'].mean())
        results_dict['cindex_std'].append(csv_df['c_index'].std())

        if len(csv_df) < 500:
            results_dict['success'].append(0)
        else:
            results_dict['success'].append(1)

results_df = pd.DataFrame(results_dict)
results_df.to_excel(f'{cv_experiment}/bootstrap_aggregated_results.xlsx')

df = results_df[results_df['success']==1]
pivot_mean = df.pivot(index='cancer_type', columns='event_type', values='cindex_mean')
pivot_std = df.pivot(index='cancer_type', columns='event_type', values='cindex_std')

# Format the cells
formatted_df = pivot_mean.applymap("{:.3f}".format) + " (" + pivot_std.applymap("{:.2f}".format) + ")"

# Calculate the mean for each column and append it to the DataFrame
mean_row = pivot_mean.mean().apply("{:.3f}".format)
formatted_df.loc['Average'] = mean_row

# Save the DataFrame to an Excel file
formatted_df.to_excel(f'{cv_experiment}/bootstrap_summary_table.xlsx')

# latex_table = formatted_df.to_latex()
# with open(f'{cv_experiment}/bootstrap_summary_table.tex', "w") as file:
#     file.write(latex_table)