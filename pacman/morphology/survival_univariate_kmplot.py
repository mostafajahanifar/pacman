import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.offsetbox import AnchoredText

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR, SURV_CANCERS
from pacman.survival.utils import add_at_risk_counts

print(7 * "=" * 7)
print("Running Survival Analysis for Morphological features in Mitotic Hot/Cold Groups")
print(7 * "=" * 7)


# Function to find the best separation point using Gaussian Mixture Model
def median_cut_off(data):
    return np.median(data)


save_root = f"{RESULTS_DIR}/morphology/kmplots/"

# keep only columns that are related to mutations
selected_feats = ["AFW"]
feat_name = selected_feats[0]
# Create a color palette
cluster_colors = {
    "MHAH": "brown",
    "MHAL": "orange",
    "MCAH": "skyblue",
    "MCAL": "royalblue",
}

censor_at = 120  # censoring time in months

# read mitosis features
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))

for cancer_type in SURV_CANCERS + ["Pan-cancer"]:  # ALL_CANCERS +
    print(cancer_type)

    save_dir = f"{save_root}/{cancer_type}"
    os.makedirs(save_dir, exist_ok=True)

    for event_type in ["OS", "PFI", "DSS", "DFI"]:
        event_time = f"{event_type}.time"

        for temp in ["Hot", "Cold"]:
            if cancer_type == "Pan-cancer":
                mitosis_feats_cancer = mitosis_feats[
                    mitosis_feats["type"].isin(ALL_CANCERS)
                ]
            else:
                mitosis_feats_cancer = mitosis_feats[
                    mitosis_feats["type"] == cancer_type
                ]
            # keep only mitotic-temp cases
            mitosis_feats_cancer = mitosis_feats_cancer[
                mitosis_feats_cancer["temperature"] == temp
            ]

            working_df = mitosis_feats_cancer[selected_feats + [event_type, event_time]]
            working_df = working_df.dropna(axis=0, how="any")

            # Reformat time events
            working_df = working_df.reset_index(drop=True)
            working_df[event_time] = working_df[event_time] / 30
            # Censor at 10 years
            ids = working_df[working_df[event_time] > censor_at].index
            working_df.loc[ids, event_time] = censor_at
            working_df.loc[ids, event_type] = 0

            if working_df[event_type].sum(axis=0) < 2:
                print(working_df[event_type].sum(axis=0))
                continue

            try:
                cutoff = median_cut_off(
                    working_df[selected_feats[0]].values.reshape(-1, 1),
                )
            except:
                continue

            if np.isnan(cutoff):
                print(f"Nan Med: {temp} -- {cancer_type}")
                continue

            # working_df["labeled_cluster"] = working_df[selected_feats[0]].apply(lambda x: f"{temp} High" if x > cutoff else f"{temp} Low")
            working_df["labeled_cluster"] = working_df[selected_feats[0]].apply(
                lambda x: f"M{temp[0]}AH" if x > cutoff else f"M{temp[0]}AL"
            )

            clusters = list(working_df["labeled_cluster"].unique())

            if len(working_df["labeled_cluster"].unique()) < 2:
                print("Less than 2 clusters found, skip this one")
                continue

            # Plot Kaplan-Meier survival plots
            kmf = KaplanMeierFitter()
            fig = plt.figure(figsize=(2, 2))

            # Store Kaplan-Meier plots to add at risk counts later
            kmf_fits = []
            for cluster in clusters:
                mask = working_df["labeled_cluster"] == cluster
                kmf.fit(
                    working_df.loc[mask, event_time],
                    event_observed=working_df.loc[mask, event_type],
                    label=cluster,
                )
                ax = kmf.plot(ci_show=False, color=cluster_colors[cluster])
                kmf_fits.append(deepcopy(kmf))

            # Perform pairwise log-rank tests and annotate p-values
            p_values = {}
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]
                    mask1 = working_df["labeled_cluster"] == cluster1
                    mask2 = working_df["labeled_cluster"] == cluster2
                    results = logrank_test(
                        working_df.loc[mask1, event_time],
                        working_df.loc[mask2, event_time],
                        event_observed_A=working_df.loc[mask1, event_type],
                        event_observed_B=working_df.loc[mask2, event_type],
                    )
                    p_values[(cluster1, cluster2)] = results.p_value
            # Open the file in write mode
            with open(
                f"{save_dir}/km_{event_type}_{selected_feats}_censor{censor_at}.txt",
                "w",
            ) as f:
                for cats, pval in p_values.items():
                    print(f"{cats[0]} vs {cats[1]}: p-value={pval:.02}", file=f)

            corrected_p_value = p_values[(cluster1, cluster2)]
            # adding the corrected p-value to the figure
            if corrected_p_value < 0.0001:
                pvalue_txt = "p < 0.0001"
            else:
                pvalue_txt = "p = " + str(np.round(corrected_p_value, 4))
            pvalue_loc = "upper right" if cancer_type == "GBMLGG" else "lower left"
            ax.add_artist(AnchoredText(pvalue_txt, loc=pvalue_loc, frameon=False))

            plt.xlim([0, 120])
            ax = plt.gca()
            if cancer_type == "Pan-cancer":
                plt.xlabel("Time (Months)")
                plt.ylabel(f"{event_type} Probability")
            else:
                plt.xlabel("")
                plt.ylabel("")
                ax.set_xticklabels([])
                ax.set_yticks([0, 0.5, 1])
                ax.set_yticklabels([])

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Position legend outside and to the right of the plot box
            plt.legend()
            ax.legend().set_visible(False)
            ax.set_title(cancer_type)
            ax.set_ylim([0, 1])
            plt.tight_layout()
            plt.savefig(
                f"{save_dir}/{temp}_km_{event_type}_{feat_name}_censor{censor_at}.png",
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.01,
            )

            # add the risk counts and plot
            fig.set_size_inches(2.2, 1.5)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels([0, 25, 50, 75, 100])
            ax.set_yticks([0, 0.5, 1])
            plt.xlabel("Time (Months)")
            plt.ylabel(f"{event_type} Probability")

            add_at_risk_counts(
                *kmf_fits,
                rows_to_show=["At risk"],
                xticks=[0, 25, 50, 75, 100],
                ypos=-0.4,
                colors=[cluster_colors[cluster] for cluster in clusters],
                ax=ax,
            )

            plt.savefig(
                f"{save_dir}/risked_{temp}_km_{event_type}_{feat_name}_censor{censor_at}.png",
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.01,
            )
