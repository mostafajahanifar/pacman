import os, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import stats
from pacman.utils import featre_to_tick, get_colors_dict
import argparse
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import linkage, leaves_list

save_root = "results_final_all/gene/methylation"
df = pd.read_csv("gene/data/data_methylation.txt", sep="\t")
df['NAME'] = df.apply(lambda row: row['ENTITY_STABLE_ID'] if pd.isna(row['NAME']) else row['NAME'], axis=1)
id_to_name = df.set_index('ENTITY_STABLE_ID')['NAME'].to_dict()



ALL_CANCERS = [
    'SARC',
    'LIHC',
    'THYM',
    # 'ACC',
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
selected_feats = [
    "HSC",
    "mean(ND)",
    "cv(ND)",
    "mit_nodeDegrees_per99",
    "mean(CL)",
    "mit_clusterCoff_std",
    "mit_clusterCoff_per90",
    "mean(HC)",
    "mit_cenHarmonic_std",
    "mit_cenHarmonic_per99",
    "mit_cenHarmonic_per10",
]

ci = 1
for i, cancer_type in enumerate(["COADREAD"]): # 
    try:
        print(f"Working on {cancer_type}")
        save_dir = f"{save_root}/{cancer_type.upper()}/"

        if cancer_type == "Pan-cancer":
            top_n = 10
        else:
            top_n = 4

        corr_r_matrix = pd.read_csv(save_dir + "corr_r.csv")
        corr_p_matrix = pd.read_csv(save_dir + "corr_p.csv")

        corr_r_matrix = corr_r_matrix.rename(columns={"Unnamed: 0":""})
        corr_p_matrix = corr_p_matrix.rename(columns={"Unnamed: 0":""})

        corr_r_matrix = corr_r_matrix.set_index(corr_r_matrix.columns[0])
        corr_p_matrix = corr_p_matrix.set_index(corr_p_matrix.columns[0])

        # corr_r_matrix = corr_r_matrix.set_index("Unnamed: 0")
        # corr_p_matrix = corr_p_matrix.set_index("Unnamed: 0")


        # Make correlation of non-significant mutations equal to zero
        corr_r_matrix_rank = corr_r_matrix.copy()
        corr_r_matrix_rank[corr_p_matrix > 0.05] = 0
        corr_r_matrix_rank = corr_r_matrix_rank.T
        corr_r_matrix = corr_r_matrix.T
        corr_p_matrix = corr_p_matrix.T

        # Create annotation matrix for significance
        annot_matrix = corr_p_matrix.applymap(lambda x: '*' if x < 0.05 else '')

        # Plot the top 20
        if len(corr_r_matrix) > 20:
            aucs_sorted = corr_r_matrix_rank.abs().max(axis=1).sort_values(ascending=False)
            max_ass = aucs_sorted.head(20).index
            corr_r_matrix = corr_r_matrix.loc[list(max_ass), :]
            annot_matrix = annot_matrix.loc[list(max_ass), :]  # Update annotation matrix for top 20
            corr_r_matrix_rank = corr_r_matrix_rank.loc[list(max_ass), :]


        # # renaming methylation codes with their names
        corr_r_matrix_rank = corr_r_matrix_rank.rename(index=id_to_name)
        corr_r_matrix = corr_r_matrix.rename(index=id_to_name)
        annot_matrix = annot_matrix.rename(index=id_to_name)

        # Perform hierarchical clustering
        row_linkage = linkage(corr_r_matrix, method='ward')
        col_linkage = linkage(corr_r_matrix.T, method='ward')

        # Get the order of rows and columns based on clustering
        row_order = leaves_list(row_linkage)
        col_order = leaves_list(col_linkage)

        # Reorder the data matrix and annotation matrix
        corr_r_matrix_rank_reordered = corr_r_matrix_rank.iloc[:, col_order].iloc[row_order, :]
        corr_r_matrix_reordered = corr_r_matrix.iloc[:, col_order].iloc[row_order, :]
        annot_matrix_reordered = annot_matrix.iloc[:, col_order].iloc[row_order, :]

        # Plot the heatmap with reordered data, annotation, and customization
        plt.figure(figsize=(4, 4))
        heatmap = sns.heatmap(corr_r_matrix_reordered, cmap="coolwarm", vmin=-1, vmax=1, cbar=False,
                            linewidths=0.5, linecolor='gray', square=True,
                            annot=annot_matrix_reordered, fmt='', annot_kws={"size": 10, "va": "center_baseline", "ha": "center"},
                            cbar_kws={'shrink': 0.5, 'label': r"Spearman's $\rho$"})

        for _, spine in heatmap.spines.items():
            spine.set_visible(True)

        plt.savefig(save_dir + f"mut_{cancer_type}_top20.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)

        # Plot max-top 4
        corr_r_matrix_reordered = corr_r_matrix_reordered[["HSC", "mean(ND)", "cv(ND)", "mean(CL)", "mean(HC)"]]
        annot_matrix_reordered = annot_matrix_reordered[["HSC", "mean(ND)", "cv(ND)", "mean(CL)", "mean(HC)"]]  # Update annot matrix for top 4

        if len(corr_r_matrix_reordered) > top_n:
            max_ass = corr_r_matrix_rank_reordered.abs().max(axis=1).sort_values(ascending=False)
            max_ass = max_ass[max_ass > 0.2]
            max_ass = max_ass.head(top_n).index
            corr_r_matrix_reordered = corr_r_matrix_reordered.loc[list(max_ass), :]
            annot_matrix_reordered = annot_matrix_reordered.loc[list(max_ass), :]  # Update annot matrix for filtered rows

        # Calculate figure size dynamically based on the number of columns (heatmap cells width)
        n_cols = corr_r_matrix_reordered.shape[0]
        print(n_cols)
        cell_size = 0.25  # size of each cell in inches
        fig_width = n_cols * cell_size
        fig_height = 1.2  # fixed height for consistency

        # Create a barplot above the heatmap
        fig, ax_heatmap = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        # Heatmap with annotations
        print(corr_r_matrix_reordered.T)
        print(annot_matrix_reordered.T)
        sns.heatmap(corr_r_matrix_reordered.T, cmap="coolwarm", vmin=-1, vmax=1, cbar=False,
                    linewidths=0.5, linecolor='gray', xticklabels=True, annot=annot_matrix_reordered.T, fmt='', annot_kws={"size": 10, "va": "center_baseline", "ha": "center"},
                    cbar_kws={'shrink': 0.5, 'label': r"Spearman's $\rho$"}, ax=ax_heatmap)

        for _, spine in ax_heatmap.spines.items():
            spine.set_visible(True)

        if ci != 0:
            plt.yticks([])

        if cancer_type == "All":
            plt.title("Pan-cancer")
        else:
            plt.title(cancer_type)

        # plt.tight_layout()
        fig.savefig(save_dir + f"mut_{cancer_type}_top5.png", dpi=600, bbox_inches='tight', pad_inches=0.01, transparent=True)
        # fig.savefig(save_dir + f"mut_{cancer_type}_top5.pdf", dpi=300, bbox_inches='tight', pad_inches=0.01)
        ci += 1
    except Exception as e:
        print(f"Error in {cancer_type}: {e}")