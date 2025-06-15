"""This code also plots surival of immune hot and immune cold groups, however, it does it based on pre-calculated
immune hot/cold groups, which are retrieved using Gaussian Mixture Model fitting on T Cells CD8
"""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from pacman.config import DATA_DIR, RESULTS_DIR, SURV_CANCERS
from pacman.survival.utils import add_at_risk_counts

print(7 * "=" * 7)
print("Running Survival Analysis for Immune Hot/Cold Groups")
print(7 * "=" * 7)

save_root = f"{RESULTS_DIR}/immune/kmplots/"
os.makedirs(save_root, exist_ok=True)


immune_feat = "T_Cells_CD8_temperature"

# Reading the immune data
immune_df = pd.read_csv(os.path.join(DATA_DIR, "tcga_all_immune_new.csv"))
immune_df = immune_df.dropna(axis=1, how="all")
# read mitosis features
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]]


for cancer_type in SURV_CANCERS:
    print(cancer_type)
    mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"] == cancer_type]

    # Find the common case names between mitosis features and gene expressions
    common_cases = pd.Series(
        list(
            set(mitosis_feats_cancer["bcr_patient_barcode"]).intersection(
                set(immune_df["TCGA Participant Barcode"])
            )
        )
    )
    ## Keep only the rows with the common case names in both dataframes
    df1_common = mitosis_feats_cancer[
        mitosis_feats_cancer["bcr_patient_barcode"].isin(common_cases)
    ]
    df2_common = immune_df[immune_df["TCGA Participant Barcode"].isin(common_cases)]
    df2_common = df2_common.drop_duplicates(subset="TCGA Participant Barcode")

    ## Sort the dataframes based on 'case_name'
    df1_common = df1_common.sort_values("bcr_patient_barcode")
    df2_common = df2_common.sort_values("TCGA Participant Barcode")

    df1_common = df1_common.reset_index(drop=True)
    df2_common = df2_common.reset_index(drop=True)

    save_dir = f"{save_root}/{cancer_type}"
    os.makedirs(save_dir, exist_ok=True)

    censor_at = 120
    for event_type in ["OS", "PFI", "DSS", "DFI"]:
        try:
            event_time = f"{event_type}.time"

            # Prepare the data (as per your code snippet)
            if event_type not in df2_common.columns:
                print(f"Event type {event_time} is not present for {cancer_type}")
                continue
            working_df = pd.concat(
                [
                    df1_common["temperature"],
                    df2_common[[immune_feat, event_type, event_time]],
                ],
                axis=1,
            )
            working_df = working_df.dropna(axis=0, how="any")

            # Reformat time events
            working_df = working_df.reset_index(drop=True)
            working_df[event_time] = working_df[event_time] / 30
            # Censor at 10 years
            ids = working_df[working_df[event_time] > censor_at].index
            working_df.loc[ids, event_time] = censor_at
            working_df.loc[ids, event_type] = 0

            def label_cluster(row):
                if row["temperature"] == "Hot" and row[immune_feat] == "Hot":
                    return "MHIH"
                elif row["temperature"] == "Hot" and row[immune_feat] == "Cold":
                    return "MHIC"
                elif row["temperature"] == "Cold" and row[immune_feat] == "Hot":
                    return "MCIH"
                elif row["temperature"] == "Cold" and row[immune_feat] == "Cold":
                    return "MCIC"

            # Apply the labeling function to create the labeled_cluster column
            working_df["labeled_cluster"] = working_df.apply(label_cluster, axis=1)
            working_df = working_df.sort_values(by="labeled_cluster")
            # Create a color palette
            clusters = ["MCIC", "MCIH", "MHIC", "MHIH"]
            colors = ["lime", "darkgreen", "hotpink", "darkorchid"]
            # Create a dictionary to map clusters to colors
            cluster_colors = dict(zip(clusters, colors))

            # Plot Kaplan-Meier survival plots
            kmf = KaplanMeierFitter()
            fig = plt.figure(figsize=(2, 2))  ##adjust according to font size

            # Store each fitted Kaplan-Meier object in a list
            kmf_fits = []

            # Store Kaplan-Meier plots to add at risk counts later
            for cluster in clusters:
                mask = working_df["labeled_cluster"] == cluster
                kmf.fit(
                    working_df.loc[mask, event_time],
                    event_observed=working_df.loc[mask, event_type],
                    label=cluster,
                )
                ax = kmf.plot(ci_show=False, color=cluster_colors[cluster])
                kmf_fits.append(
                    deepcopy(kmf)
                )  # Append the fitted kmf object to the list to be used later

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
                f"{save_dir}/km_{event_type}_{immune_feat}_censor{censor_at}.txt", "w"
            ) as f:
                for cats, pval in p_values.items():
                    print(f"{cats[0]} vs {cats[1]}: p-value={pval:.02}", file=f)

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
            plt.legend(title=f"Immune: {immune_feat}")
            ax.legend().set_visible(False)
            ax.set_title(cancer_type)
            ax.set_ylim([0, 1])
            plt.tight_layout()
            plt.savefig(
                f"{save_dir}/km_{event_type}_{immune_feat}_censor{censor_at}.png",
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
                colors=colors,
                ax=ax,
            )

            plt.savefig(
                f"{save_dir}/risked_km_{event_type}_{immune_feat}_censor{censor_at}.png",
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.01,
            )

            # Map each row to a color
            working_df["temperature"] = working_df["temperature"].apply(
                lambda x: 1 if x == "Hot" else 0
            )
            working_df[immune_feat] = working_df[immune_feat].apply(
                lambda x: 1 if x == "Hot" else 0
            )
            row_colors = working_df["labeled_cluster"].map(cluster_colors)
            row_colors = row_colors.to_list()
            working_df = working_df.rename(
                columns={"temperature": "Mitosis", immune_feat: "Immune"}
            )
            # Plot the clustermap
            g = sns.clustermap(
                working_df[["Mitosis", "Immune"]],
                standard_scale=1,
                z_score=None,
                col_cluster=False,
                cmap="coolwarm",
                row_cluster=False,
                method="ward",
                figsize=(0.6, 2),
                cbar_pos=None,
                yticklabels=False,
                xticklabels=False,
                row_colors=row_colors,
                dendrogram_ratio=0,
                colors_ratio=0.2,
            )

            # # Add vertical lines between columns using axvline
            for i in range(
                1, working_df[["Mitosis", "Immune"]].shape[1]
            ):  # Loop over columns
                g.ax_heatmap.axvline(
                    i, color="white", lw=0.2
                )  # Add vertical linecbar_{immune_feat}.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
            plt.savefig(
                f"{save_dir}/cbar_{event_type}_{immune_feat}.png",
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.01,
            )
        except Exception as e:
            print(f"Error processing {cancer_type} and {event_type}: {e}")
