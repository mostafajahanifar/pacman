import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from utils import featre_to_tick

# Set seaborn style for better aesthetics
# sns.set(style="whitegrid")

censor_at = 120
cv_experiment = 'CV_results_10years_median'


event_types = ['DFI', 'PFI', 'OS', 'DSS']
for event_type in event_types:
    cancer_types =  [["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"], ["GBM"], ["HNSC"], ["KICH"], ["KIRC"], ["KIRP"], ["LGG"], ["LIHC"], ["LUAD"], ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"]]
    for cancer_type in cancer_types:
        print(event_type, cancer_type)
        save_results_path = f'features_hazard_visualized/{cv_experiment}/{event_type}/'
        os.makedirs(save_results_path, exist_ok=True)

        csv_path = f"{cv_experiment}/CV_{cancer_type}/bootstrap_results_{cancer_type}_{event_type}_censor{censor_at}.csv"

        try:
            exp_df = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f'No bootstrap results available for {cancer_type} - {event_type}')
            continue

        if len(exp_df) < 500:
            print(f'Unsuccessful bootstrap analysis for {cancer_type} - {event_type}')
            continue
        
        # Handle NaN and infinite values
        exp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        exp_df.dropna(inplace=True)

        ff = set(exp_df.columns) - set(['c_index', 'p_value'])
        exp_df.columns = exp_df.columns.str.replace(r'^mit_', '', regex=True)
        features = set(exp_df.columns) - set(['c_index', 'p_value'])
        
        # Create violin plot, Fayyaz way
        plt.figure(figsize=(10, len(ff) * 0.5))
        cphHR = exp_df[list(features)].to_numpy()
        ww = np.median(cphHR,axis=0)
        idx = np.arange(len(ff)) # np.argsort(np.abs(ww))
        ffx = np.array([featre_to_tick(f) for f in ff])
        plt.violinplot(np.log(cphHR)[:,idx[::-1]],showmedians=True,showextrema=True,vert=False,widths=0.9)
        plt.yticks(list(np.arange(1,len(ff)+1)), ffx[idx[::-1]])
        plt.yticks(rotation=0)
        plt.title(f"{''.join(cancer_type).upper()} - {event_type}")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('log(HR)',fontsize=16)

        # # Create violin plot
        # plt.figure(figsize=(10, len(features) * 0.5))  # Adjust the figure height based on the number of features
        # try:
        #     ax = sns.violinplot(data=exp_df[list(features)], inner=None, width=0.8, orient='h', color="skyblue", linewidth=0)  # Set orient='h' for horizontal
        # except:
        #     print('Failed in :', csv_path)
        #     continue
        #     # ax = sns.violinplot(data=exp_df[list(features)], inner=None, width=0.8, orient='h', color="skyblue", linewidth=0, bw=0.1)
        # # Set plot labels and title
        # ax.set(xlabel="Hazard Ratios", ylabel="Features", title=f"{''.join(cancer_type).upper()} - {event_type}")
        # ax.tick_params(axis='y', rotation=20)  # Rotate y-axis ticks
        # if cancer_type != ["KICH"]:
        #     ax.set_xlim(-1, 6)
        # else:
        #     ax.set_xlim(-1, 10)

        # Save the figure with high DPI and without strokes
        plt.savefig(f"{save_results_path}/{cancer_type}_{event_type}.png", bbox_inches='tight', dpi=600)
        plt.close()
