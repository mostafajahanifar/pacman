import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import featre_to_tick

event_types = ['DFI', 'PFI', 'DSS', 'OS']
features = [
"mit_nodeDegrees_mean",
"mit_nodeDegrees_cv",
"mit_nodeDegrees_perc99",
"mit_clusterCoff_mean",
"mit_clusterCoff_std",
"mit_clusterCoff_perc90",
"mit_cenHarmonic_mean",
"mit_cenHarmonic_std",
"mit_cenHarmonic_perc10",
"mit_cenHarmonic_perc99",
]

all_cancer_types = [
    ["ACC"], ["BLCA"], ["BRCA"], ["CESC"], ["COAD", "READ"], ["ESCA"],
    ["GBM", "LGG"], ["HNSC"], ["KIRC"], ["KIRP"], ["LIHC"], ["LUAD"],
    ["LUSC"], ["OV"], ["PAAD"], ["SKCM"], ["STAD"], ["UCEC"], ["MESO"],
    ["PRAD"], ["SARC"], ["TGCT"], ["THCA"]
]

cv_experiment = 'results_final/survival/bootstrap_10years_feat10sel/'
censor_at = 120

for event_type in event_types:
    print("**************", event_type)
    data = {}
    cancer_types_names = []
    for cancer_type in all_cancer_types:
        cancer_str = ''.join(cancer_type)
        csv_path = f"{cv_experiment}/CV_{cancer_type}/bootstrap_results_{cancer_type}_{event_type}_censor{censor_at}.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            data[cancer_str] = df
            cancer_types_names.append(cancer_str)
        else:
            print(f"File not found: {csv_path}")
            data[cancer_str] = None

    cancer_types = [cancer_str for cancer_str in data if data[cancer_str] is not None]
    num_features = len(features)
    num_cancer_types = len(cancer_types)
    
    # creating figure canvas
    width_per_subplot = 1  # Example: 5 inches per subplot width
    height_per_subplot = width_per_subplot / 3.5  # Height is 1/5th of width
    # Calculate the total figure size
    fig_width = num_features * width_per_subplot
    fig_height = num_cancer_types * height_per_subplot
    # Create the subplots with the calculated figure size
    fig, axs = plt.subplots(num_cancer_types, num_features, figsize=(fig_width, fig_height), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.05)
    
    for fi in range(num_features):
        for ci in range(num_cancer_types):
            # if cancer_types_names[ci] not in valid_cancer_types:
            #     continue
            if features[fi] not in data[cancer_types_names[ci]].columns:
                continue
            # print(f"Plotting {features[fi]} for {cancer_types_names[ci]}")
            axs[ci, fi].violinplot(data[cancer_types_names[ci]][features[fi]], vert=False, widths=1, showmedians=False, showextrema=True,)
    
    for fi in range(num_features):
        for ci in range(num_cancer_types):
            axs[ci, fi].spines['top'].set_visible(False)
            axs[ci, fi].spines['right'].set_visible(False)
            axs[ci, fi].axhline(y=1, color='lightgray', linestyle='--', zorder=0, linewidth=1)
            axs[ci, fi].axvline(x=1, color='lightgray', zorder=0, linewidth=1)

            if ci != num_cancer_types-1:
                axs[ci, fi].spines['bottom'].set_visible(False)
                axs[ci, fi].set_xticks([])
            
            if fi == 0:  # Only show left boundary on the leftmost plots
                axs[ci, fi].spines['left'].set_visible(True)
            else:
                axs[ci, fi].spines['left'].set_visible(False)

            if ci == num_cancer_types - 1:
                axs[ci, fi].set_xlabel(features[fi])

            if ci != num_cancer_types - 1:
                axs[ci, fi].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            
            axs[ci, fi].set_xlim([-0.1, 3.2])
            
                
    # put cancer names
    for ci in range(num_cancer_types):
        axs[ci,0].set_yticklabels('')
        axs[ci,0].set_yticks([])
        axs[ci,0].set_ylabel(cancer_types_names[ci], rotation=0, ha='right', va='center', fontsize=10)

    # put feature names
    for fi in range(num_features):
        axs[0,fi].set_title(featre_to_tick(features[fi]), rotation=0, fontsize=10)
        axs[-1,fi].set_xlabel("HR", rotation=0, fontsize=10)
        axs[-1, fi].set_xticks([0, 1, 3])
    
    plt.subplots_adjust(wspace=0.1, hspace=0)
    print("Completed")
    plt.savefig(f"{cv_experiment}/bootstrap_hr_{event_type}_censor{censor_at}.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f"{cv_experiment}/bootstrap_hr_{event_type}_censor{censor_at}.pdf", dpi=600, bbox_inches='tight', pad_inches=0.01)