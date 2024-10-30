"""This code also plots surival of immune hot and immune cold groups, however, it does it based on pre-calculated
immune hot/cold groups, which are retrieved using Gaussian Mixture Model fitting on T Cells CD8"""
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


save_root = "results_final_all/immune/survival_2/"

IMPORTANT_CANCERS = ["ACC", "BLCA", "BRCA", "CESC", "COADREAD", "ESCA", "GBMLGG", "HNSC", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "OV", "PAAD", "SKCM", "STAD", "UCEC", "MESO", "PRAD", "SARC", "TGCT", "THCA", "KICH"]
ALL_CANCERS = ['SARC',
    'LIHC',
    'THYM',
    'ACC',
    'BRCA',
    'KICH',
    'STAD',
    'BLCA',
    'THCA',
    'GBMLGG',
    'UCEC',
    'LUAD',
    'KIRC',
    'KIRP',
    'PAAD',
    'CESC',
    'PCPG',
    'MESO',
    'SKCM',
    'PRAD',
    'COADREAD',
    'ESCA',
    'LUSC',
    'HNSC',
    'OV',
    'TGCT',
    'CHOL',
    'DLBC',
    'UCS'
 ]
ALL_CANCERS = sorted(ALL_CANCERS)

# keep only columns that are related to mutations
immune_df = pd.read_csv("gene/data/tcga_all_immune_new.csv")
immune_df["TCGA Study"] = immune_df["TCGA Study"].replace(["COAD", "READ"], "COADREAD")
immune_df["TCGA Study"] = immune_df["TCGA Study"].replace(["GBM", "LGG"], "GBMLGG")
# immune_df = immune_df[["TCGA Participant Barcode", "TCGA Study", "T_Cells_CD8_temperature"]]

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

mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancer_withAtypical.csv')
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]+selected_feats]
mitosis_feats.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type", "temperature"] else col for col in mitosis_feats.columns]
mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")

for cancer_type in  IMPORTANT_CANCERS:# ["Pan-cancer"]: # IMPORTANT_CANCERS +
    print(cancer_type)
    if cancer_type == "Pan-cancer":
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

    immune_feat =  "T_Cells_CD8_temperature"
    censor_at = 120
    for event_type in ["OS", "PFI", "DSS", "DFI"]:
        event_time = f"{event_type}.time"

        # Prepare the data (as per your code snippet)
        if event_type not in df2_common.columns:
            print(f"Event type {event_time} is not present for {cancer_type}")
            continue
        mosi = pd.concat([df1_common["temperature"], df2_common[[immune_feat, event_type, event_time]]], axis=1)
        mosi = mosi.dropna(axis=0, how="any")

        # Reformat time events
        mosi = mosi.reset_index(drop=True)
        mosi[event_time] = mosi[event_time] / 30
        # Censor at 10 years
        ids = mosi[mosi[event_time] > censor_at].index
        mosi.loc[ids, event_time] = censor_at
        mosi.loc[ids, event_type] = 0

        def label_cluster(row):
            if row['temperature'] == "Hot" and row[immune_feat] == "Hot":
                return 'Mitotic-Hot, Immune-Hot'
            elif row['temperature'] == "Hot" and row[immune_feat] == "Cold":
                return 'Mitotic-Hot, Immune-Cold'
            elif row['temperature'] == "Cold" and row[immune_feat] == "Hot":
                return 'Mitotic-Cold, Immune-Hot'
            elif row['temperature'] == "Cold" and row[immune_feat] == "Cold":
                return 'Mitotic-Cold, Immune-Cold'

        # Apply the labeling function to create the labeled_cluster column
        mosi['labeled_cluster'] = mosi.apply(label_cluster, axis=1)
        mosi = mosi.sort_values(by='labeled_cluster')
        # Create a color palette
        cluster_colors = {"Mitotic-Hot, Immune-Cold": "hotpink",
                        "Mitotic-Hot, Immune-Hot": "darkorchid",
                        "Mitotic-Cold, Immune-Cold": "lime",
                        "Mitotic-Cold, Immune-Hot": "darkgreen"}
        unique_clusters = list(cluster_colors.keys())

        # Plot Kaplan-Meier survival plots
        kmf = KaplanMeierFitter()
        plt.figure(figsize=(2, 2))

        # Store Kaplan-Meier plots to add at risk counts later
        clusters = mosi['labeled_cluster'].unique()
        for cluster in clusters:
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
        ax = plt.gca()
        if cancer_type == "Pan-cancer":
            plt.xlabel('Time (Months)')
            plt.ylabel(f'{event_type} Probability')
        else:
            plt.xlabel('')
            plt.ylabel('')
            ax.set_xticklabels([])
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels([])
            
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
        mosi["temperature"] = mosi["temperature"].apply(lambda x: 1 if x == "Hot" else 0)
        mosi[immune_feat] = mosi[immune_feat].apply(lambda x: 1 if x == "Hot" else 0)
        row_colors = mosi['labeled_cluster'].map(cluster_colors)
        row_colors = row_colors.to_list()
        mosi = mosi.rename(columns={"temperature":"Mitosis", immune_feat:"Immune"})
        # Plot the clustermap
        g = sns.clustermap(mosi[["Mitosis", "Immune"]], standard_scale=1, z_score=None, col_cluster=False, cmap="coolwarm",
                    row_cluster=False, method='ward', figsize=(0.6, 2), cbar_pos=None, yticklabels=False, xticklabels=False,
                    row_colors=row_colors, dendrogram_ratio=0, colors_ratio=0.2)

        # # Add vertical lines between columns using axvline
        for i in range(1, mosi[["Mitosis", "Immune"]].shape[1]):  # Loop over columns
            g.ax_heatmap.axvline(i, color='white', lw=0.2)  # Add vertical linecbar_{immune_feat}.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
        plt.savefig(f"{save_dir}/cbar_{event_type}_{immune_feat}.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)

        if add_counts:
            fig_copy = plt.figure(figsize=(fig_size, fig_size-2))
            ax_copy = fig_copy.add_subplot(111)
            ax_copy.set_xlabel('', fontsize=font_size)
            ax_copy.set_ylabel('', fontsize=font_size)
            ax_copy.tick_params(axis='x', labelsize=font_size)
            ax_copy.tick_params(axis='y', labelsize=font_size)

            # Initializing the KaplanMeierModel for each group
            ax_copy = km_upper.fit(T_upper_test, event_observed=E_upper_test, label='high').plot_survival_function(ax=ax_copy, show_censors=True, censor_styles={'ms': 5}, color='r', ci_show=False, xlabel=x_label, ylabel=y_label)
            ax_copy = km_lower.fit(T_lower_test, event_observed=E_lower_test, label='low').plot_survival_function(ax=ax_copy, show_censors=True, censor_styles={'ms': 5}, color='b', ci_show=False, xlabel=x_label, ylabel=y_label)

            add_at_risk_counts(km_upper, km_lower, ax=ax_copy, fig=fig_copy, fontsize=int(font_size*1))
            fig_copy.subplots_adjust(bottom=0.4)
            fig_copy.subplots_adjust(left=0.2)
            ax_copy.get_legend().remove()