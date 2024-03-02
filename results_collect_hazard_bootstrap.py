import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# Set seaborn style for better aesthetics
# sns.set(style="whitegrid")

censor_at = -1
cv_experiment = 'CV_results_NoCensor_median'


event_types = ['DFI', 'PFI', 'OS', 'DSS']
for event_type in event_types:
    cancer_types = [["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"], ["GBM"], ["HNSC"], ["KICH"], ["KIRC"], ["KIRP"], ["LGG"], ["LIHC"], ["LUAD"], ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"]]

    for cancer_type in cancer_types:
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

        exp_df.columns = exp_df.columns.str.replace(r'^mit_', '', regex=True)
        features = set(exp_df.columns) - set(['c_index', 'p_value'])
        

        # Create violin plot
        plt.figure(figsize=(10, len(features) * 0.5))  # Adjust the figure height based on the number of features
        try:
            ax = sns.violinplot(data=exp_df[list(features)], inner=None, width=0.8, orient='h', color="skyblue", linewidth=0)  # Set orient='h' for horizontal
        except:
            print('Failed in :', csv_path)
            continue
            # ax = sns.violinplot(data=exp_df[list(features)], inner=None, width=0.8, orient='h', color="skyblue", linewidth=0, bw=0.1)

        # Set plot labels and title
        ax.set(xlabel="Hazard Ratios", ylabel="Features", title=f"{''.join(cancer_type).upper()} - {event_type}")
        ax.tick_params(axis='y', rotation=20)  # Rotate y-axis ticks
        if cancer_type != ["KICH"]:
            ax.set_xlim(-1, 6)
        else:
            ax.set_xlim(-1, 10)

        # Save the figure with high DPI and without strokes
        plt.savefig(f"{save_results_path}/{cancer_type}_{event_type}.png", bbox_inches='tight', dpi=600)
        plt.close()
