import os
import pandas as pd
import numpy as np
from utils import featre_to_tick
import matplotlib.pyplot as plt


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


save_root = "results_final_all/morphology/survival_WAF/"

ALL_CANCERS = ['BRCA', 'GBMLGG', 'COADREAD', 'KIRC', 'UCEC', 'LUSC', 'LUAD', 'HNSC',
       'THCA', 'SKCM', 'BLCA', 'STAD', 'LIHC', 'PRAD', 'KIRP', 'CESC', 'SARC']

selected_feats = [
    # "aty_hotspot_count",
    # "aty_hotspot_ratio",
    "aty_wsi_ratio",
]

mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancer_withAtypical.csv')
mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")

for cancer_type in  ALL_CANCERS + ["Pan-cancer"]: # 
    print(cancer_type)
    
    save_dir = f"{save_root}/{cancer_type}"
    os.makedirs(save_dir, exist_ok=True)

    censor_at = 120
    for event_type in ["OS", "PFI", "DSS", "DFI"]:
        event_time = f"{event_type}.time"

        if cancer_type == "Pan-cancer":
            mitosis_feats_cancer = mitosis_feats
        else:
            mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"] == cancer_type]

        # Keep both Hot and Cold samples in one DataFrame
        mosi = mitosis_feats_cancer[selected_feats + [event_type, event_time, "temperature"]]
        mosi = mosi.dropna(axis=0, how="any")

        # Reformat time events
        mosi = mosi.reset_index(drop=True)
        mosi[event_time] = mosi[event_time] / 30
        # Censor at 10 years
        ids = mosi[mosi[event_time] > censor_at].index
        mosi.loc[ids, event_time] = censor_at
        mosi.loc[ids, event_type] = 0

        # Compute the median for the feature and label clusters for both Hot and Cold samples
        feat_med = mosi[selected_feats].median().values[0]
        print(feat_med)
        if np.isnan(feat_med):
            print(f"Nan Med: {cancer_type}")
            continue
        mosi["labeled_cluster"] = mosi.apply(
            lambda row: f"{row['temperature']} High" if row[selected_feats[0]] > feat_med else f"{row['temperature']} Low", 
            axis=1
        )

        # Create a color palette
        cluster_colors = {
            "Hot High": "brown",
            "Hot Low": "orange",
            "Cold High": "skyblue",
            "Cold Low": "royalblue"
        }

        # Plot Kaplan-Meier survival plots for both Hot and Cold in one figure
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(3, 3))  # Increase the figure size for clarity

        # Store Kaplan-Meier plots to add at risk counts later
        for cluster in mosi['labeled_cluster'].unique():
            mask = mosi['labeled_cluster'] == cluster
            kmf.fit(mosi.loc[mask, event_time], event_observed=mosi.loc[mask, event_type], label=cluster)
            ax = kmf.plot(ci_show=False, color=cluster_colors[cluster])

        # Perform pairwise log-rank tests and annotate p-values
        p_values = {}
        clusters = mosi['labeled_cluster'].unique()
        if len(clusters) < 2:
            continue
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster1 = clusters[i]
                cluster2 = clusters[j]
                mask1 = mosi['labeled_cluster'] == cluster1
                mask2 = mosi['labeled_cluster'] == cluster2
                results = logrank_test(mosi.loc[mask1, event_time], mosi.loc[mask2, event_time],
                                       event_observed_A=mosi.loc[mask1, event_type], event_observed_B=mosi.loc[mask2, event_type])
                p_values[(cluster1, cluster2)] = results.p_value

        # Create a text file for this figure, to store colors and significant p-values
        text_file_path = f"{save_dir}/km_{event_type}_{selected_feats}_censor{censor_at}.txt"
        with open(text_file_path, 'w') as f:
            f.write("\nSignificant p-values (p < 0.05):\n")
            has_significant_pvalues = False
            # Write only significant p-values
            for (cluster1, cluster2), pval in p_values.items():
                if pval < 0.05:
                    f.write(f"{cluster1} ({cluster_colors[cluster1]}) vs {cluster2} ({cluster_colors[cluster2]}): p = {pval:.4f}\n")
                    has_significant_pvalues = True

            if not has_significant_pvalues:
                f.write("No significant p-values.\n")

        # Adjust plot limits and labels
        plt.xlim([0, 120])
        plt.xlabel('Time (Months)')
        plt.ylabel(f'{event_type} Probability')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend().set_visible(False)

        # No legend, only save the plot
        ax.set_title(cancer_type)
        ax.set_ylim([0, 1])
        plt.tight_layout()

        # Save the figure
        plt.savefig(f"{save_dir}/km_combined_{event_type}_{selected_feats}_censor{censor_at}.png", dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()
