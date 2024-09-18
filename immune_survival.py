import os, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from utils import featre_to_tick, get_colors_dict
import argparse
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm
import seaborn as sns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.cluster import AgglomerativeClustering
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.cluster import KMeans


save_root = "results_final/immune/survival/"

ALL_CANCERS = ["ACC", "BLCA", "BRCA", "CESC", "COADREAD", "ESCA", "GBMLGG", "HNSC", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "OV", "PAAD", "SKCM", "STAD", "UCEC", "MESO", "PRAD", "SARC", "TGCT", "THCA", "KICH"]


# keep only columns that are related to mutations
immune_df = pd.read_csv("gene/data/tcga_all_immune.csv")
immune_df["TCGA Study"] = immune_df["TCGA Study"].replace(["COAD", "READ"], "COADREAD")
immune_df["TCGA Study"] = immune_df["TCGA Study"].replace(["GBM", "LGG"], "GBMLGG")

selected_feats = [
"mit_hotspot_count",
"mit_nodeDegrees_mean",
"mit_nodeDegrees_cv",
"mit_nodeDegrees_per99",
"mit_clusterCoff_mean",
"mit_clusterCoff_std",
"mit_clusterCoff_per90",
"mit_cenHarmonic_mean",
"mit_cenHarmonic_std",
"mit_cenHarmonic_per10",
"mit_cenHarmonic_per99",
]

mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancer.csv')
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]+selected_feats]
mitosis_feats.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type", "temperature"] else col for col in mitosis_feats.columns]
mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")

for cancer_type in  ["All"]: # ALL_CANCERS +
    print(cancer_type)
    if cancer_type == "All":
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"].isin(ALL_CANCERS)]
        gene_exp_cancer = immune_df[immune_df["TCGA Study"].isin(ALL_CANCERS)]
    else:
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"]==cancer_type]
        gene_exp_cancer = immune_df[immune_df["TCGA Study"]==cancer_type]


    # drop missing mutations
    gene_exp_cancer = gene_exp_cancer.dropna(axis=1, how="all")
    # drop cases with all mutations as Nan


    # Find the common case names between mitosis features and gene expressions
    common_cases = pd.Series(list(set(mitosis_feats_cancer['bcr_patient_barcode']).intersection(set(gene_exp_cancer['TCGA Participant Barcode']))))
    ## Keep only the rows with the common case names in both dataframes
    df1_common = mitosis_feats_cancer[mitosis_feats_cancer['bcr_patient_barcode'].isin(common_cases)]
    df2_common = gene_exp_cancer[gene_exp_cancer['TCGA Participant Barcode'].isin(common_cases)]
    df2_common = df2_common.drop_duplicates(subset='TCGA Participant Barcode')

    ## Sort the dataframes based on 'case_name'
    df1_common = df1_common.sort_values('bcr_patient_barcode')
    df2_common = df2_common.sort_values('TCGA Participant Barcode')

    df1_common = df1_common.reset_index(drop=True)
    df2_common = df2_common.reset_index(drop=True)

    save_dir = f"{save_root}/{cancer_type}"
    os.makedirs(save_dir, exist_ok=True)

    immune_feat =  "T Cells CD8" # "Lymphocytes" #
    censor_at = 120
    for event_type in ["OS", "PFI", "DSS", "DFI"]:
        event_time = f"{event_type}.time"

        # Prepare the data (as per your code snippet)
        if event_type not in df2_common.columns:
            print(f"Event type {event_time} is not present for {cancer_type}")
            continue
        mosi = pd.concat([df1_common["temperature"], df2_common[[immune_feat, event_type, event_time]]], axis=1)
        mosi = mosi.dropna(axis=0, how="any")
        mosi["temperature"] = mosi["temperature"].apply(lambda x: 1 if x == "Hot" else 0)

        # Reformat time events
        mosi = mosi.reset_index(drop=True)
        mosi[event_time] = mosi[event_time] / 30
        # Censor at 10 years
        ids = mosi[mosi[event_time] > censor_at].index
        mosi.loc[ids, event_time] = censor_at
        mosi.loc[ids, event_type] = 0

        # Standardize the features
        mosi[["temperature", immune_feat]] = (mosi[["temperature", immune_feat]] - mosi[["temperature", immune_feat]].min()) / (mosi[["temperature", immune_feat]].max() - mosi[["temperature", immune_feat]].min())

        df = mosi.copy()

        # Split the DataFrame into two based on mitosis column
        df_mitosis_1 = df[df['temperature'] == 1].copy()
        df_mitosis_0 = df[df['temperature'] == 0].copy()

        # Apply KMeans clustering to each subset based on the immune column
        kmeans_1 = KMeans(n_clusters=2, random_state=0).fit(df_mitosis_1[[immune_feat]])
        kmeans_0 = KMeans(n_clusters=2, random_state=0).fit(df_mitosis_0[[immune_feat]])

        # Add cluster labels to the subsets
        df_mitosis_1['immune_cluster'] = kmeans_1.labels_
        df_mitosis_0['immune_cluster'] = kmeans_0.labels_

        # Function to determine which cluster is "cold" and which is "hot"
        def determine_cluster_labels(df, cluster_col, value_col):
            cluster_means = df.groupby(cluster_col)[value_col].mean()
            cold_cluster = cluster_means.idxmin()
            hot_cluster = cluster_means.idxmax()
            return {cold_cluster: 'Immune-Cold', hot_cluster: 'Immune-Hot'}

        # Determine cluster labels for mitosis 1 and mitosis 0 subsets
        cluster_labels_1 = determine_cluster_labels(df_mitosis_1, 'immune_cluster', immune_feat)
        cluster_labels_0 = determine_cluster_labels(df_mitosis_0, 'immune_cluster', immune_feat)

        # Apply the cluster labels to the subsets
        df_mitosis_1['immune_label'] = df_mitosis_1['immune_cluster'].map(cluster_labels_1)
        df_mitosis_0['immune_label'] = df_mitosis_0['immune_cluster'].map(cluster_labels_0)

        # Function to label the clusters
        def label_cluster(row):
            if row['temperature'] == 1:
                return f'Mitotic-Hot, {row["immune_label"]}'
            else:
                return f'Mitotic-Cold, {row["immune_label"]}'

        # Combine the subsets back into a single DataFrame
        df_combined = pd.concat([df_mitosis_1, df_mitosis_0], ignore_index=True)

        # Apply the labeling function to create the labeled_cluster column
        df_combined['labeled_cluster'] = df_combined.apply(label_cluster, axis=1)
        mosi = df_combined.sort_values(by='labeled_cluster')
        # Create a color palette
        cluster_colors = {"Mitotic-Hot, Immune-Cold": "hotpink",
                        "Mitotic-Hot, Immune-Hot": "darkorchid",
                        "Mitotic-Cold, Immune-Cold": "lime",
                        "Mitotic-Cold, Immune-Hot": "darkgreen"}
        unique_clusters = list(cluster_colors.keys())

        # Plot Kaplan-Meier survival plots
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(3, 3))

        # Store Kaplan-Meier plots to add at risk counts later
        for cluster in unique_clusters:
            mask = mosi['labeled_cluster'] == cluster
            kmf.fit(mosi.loc[mask, event_time], event_observed=mosi.loc[mask, event_type], label=cluster)
            ax = kmf.plot(ci_show=False, color=cluster_colors[cluster])

        # Perform pairwise log-rank tests and annotate p-values
        p_values = {}
        clusters = mosi['labeled_cluster'].unique()
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster1 = clusters[i]
                cluster2 = clusters[j]
                mask1 = mosi['labeled_cluster'] == cluster1
                mask2 = mosi['labeled_cluster'] == cluster2
                results = logrank_test(mosi.loc[mask1, event_time], mosi.loc[mask2, event_time],
                                    event_observed_A=mosi.loc[mask1, event_type], event_observed_B=mosi.loc[mask2, event_type])
                p_values[(cluster1, cluster2)] = results.p_value
        # Open the file in write mode
        with open(f"{save_dir}/km_{event_type}_{immune_feat}_censor{censor_at}.txt", 'w') as f:
            for cats, pval in p_values.items():
                print(f"{cats[0]} vs {cats[1]}: p-value={pval:.02}", file=f)


        plt.xlim([0, 120])
        plt.xlabel('Time (Months)')
        plt.ylabel(f'{event_type} Probability')
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Position legend outside and to the right of the plot box
        plt.legend(title=f"Immune: {immune_feat}")
        ax.legend().set_visible(False)
        ax.set_title(cancer_type)
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f"{save_dir}/km_{event_type}_{immune_feat}_censor{censor_at}.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)

        # Map each row to a color
        row_colors = mosi['labeled_cluster'].map(cluster_colors)
        row_colors = row_colors.rename("Group")
        mosi = mosi.rename(columns={"temperature":"Mitosis", immune_feat:"Immune"})
        # Plot the clustermap
        sns.clustermap(mosi[["Mitosis", "Immune"]], standard_scale=1, z_score=None, col_cluster=False, cmap="coolwarm",
                    row_cluster=False, method='ward', figsize=(0.7, 3), cbar_pos=None, yticklabels=False, xticklabels=True,
                    row_colors=row_colors, dendrogram_ratio=0, colors_ratio=0.2)
        plt.savefig(f"{save_dir}/cbar_{immune_feat}.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)