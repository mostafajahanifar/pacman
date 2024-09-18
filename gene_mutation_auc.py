import os, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import stats
from utils import featre_to_tick, get_colors_dict
import argparse
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from scipy.cluster.hierarchy import linkage, leaves_list

def auc_association_matrix(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with NaN values in Y and align the indices
    non_nan_indices = ~Y.isna().any(axis=1)
    X_clean = X.loc[non_nan_indices]
    Y_clean = Y.loc[non_nan_indices]
    
    # Standardize the columns of X to the 0-1 range
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
    
    # Initialize an empty DataFrame for the AUC association matrix
    auc_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)
    
    # Iterate over each combination of columns from X and Y
    for x_col in X.columns:
        for y_col in Y.columns:
            # Compute the AUC score for the current pair of columns
            auc_score = roc_auc_score(Y_clean[y_col], X_scaled[x_col])
            # Store the AUC score in the matrix
            auc_matrix.at[x_col, y_col] = 2*(auc_score - 0.5)
    
    return auc_matrix.astype(float).T

save_root = "results_final/gene/mutation"

# keep only columns that are related to mutations
gene_expr_all = pd.read_csv("gene/data/tcga_all_gene_mutations.csv")
# sel_cols = [col for col in gene_expr_all.columns if "_mut" in col]
# gene_expr_all = gene_expr_all[['type', 'case_id'] + sel_cols]
# sel_cols = [col for col in gene_expr_all.columns if "_mut" in col]
# col_rename_dict = {col: col.split("_")[0] for col in sel_cols}
# gene_expr_all = gene_expr_all.rename(columns=col_rename_dict)

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
    "mit_cenHarmonic_per99",
    "mit_cenHarmonic_per10",
]
# mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final.csv')
mitosis_feats = pd.read_csv('/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_final.csv')
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type"]+selected_feats]
mitosis_feats.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type"] else col for col in mitosis_feats.columns]
mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")

for ci, cancer_type in enumerate(sorted(gene_expr_all["type"].unique())):
    print(f"Working on {cancer_type}")
    save_dir = f"{save_root}/{cancer_type}/"
    os.makedirs(save_dir, exist_ok=True)

    mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"]==cancer_type]
    gene_exp_cancer = gene_expr_all[gene_expr_all["type"]==cancer_type]

    # drop missing mutations
    gene_exp_cancer = gene_exp_cancer.dropna(axis=1, how="all")
    # drop cases with all mutations as Nan

    # Find the common case names between mitosis features and gene expressions
    common_cases = pd.Series(list(set(mitosis_feats_cancer['bcr_patient_barcode']).intersection(set(gene_exp_cancer['case_id']))))
    ## Keep only the rows with the common case names in both dataframes
    df1_common = mitosis_feats_cancer[mitosis_feats_cancer['bcr_patient_barcode'].isin(common_cases)]
    df2_common = gene_exp_cancer[gene_exp_cancer['case_id'].isin(common_cases)]
    df2_common = df2_common.drop_duplicates(subset='case_id')

    ## Sort the dataframes based on 'case_name'
    df1_common = df1_common.sort_values('bcr_patient_barcode')
    df2_common = df2_common.sort_values('case_id')

    X = df1_common.drop(columns=["bcr_patient_barcode", "type"]).reset_index(drop=True)
    Y = df2_common.drop(columns=['case_id', 'type']).reset_index(drop=True)

    # drop genes with less than 4% mutations
    mut_thresh = int(0.04*len(Y))
    Y = Y[Y.sum(axis=0).index[Y.sum(axis=0)>mut_thresh]]

    # drop mutations with zero std (only one label)
    Y = Y[Y.std(axis=0).index[Y.std(axis=0)!=0]]


    # Measure feature-mutation association
    auc_matrix = auc_association_matrix(X, Y)
    # drop duplicated mutations
    auc_matrix = auc_matrix.loc[~auc_matrix.index.duplicated(keep='first')]
    # save aux matrix
    auc_matrix.to_csv(save_dir+f"{cancer_type}_auc_matrix.csv")

    # Plot the top 20
    if len(auc_matrix) > 20:
        aucs_sorted = auc_matrix.abs().max(axis=1).sort_values(ascending=False)
        max_ass = aucs_sorted.head(20).index
        auc_matrix = auc_matrix.loc[list(max_ass), :]

    # Perform hierarchical clustering
    row_linkage = linkage(auc_matrix, method='ward')
    col_linkage = linkage(auc_matrix.T, method='ward')

    # Get the order of rows and columns based on clustering
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    # Reorder the data matrix
    auc_matrix_reordered = auc_matrix.iloc[:, col_order].iloc[row_order, :]
    # Plot the heatmap with reordered data and customization
    plt.figure(figsize=(4, 4))
    heatmap = sns.heatmap(auc_matrix_reordered, cmap="coolwarm", vmin=-1, vmax=1, cbar=False, 
                        linewidths=0.5, linecolor='gray', square=True,
                        cbar_kws={'shrink': 0.5, 'label': 'Scaled AUC'})


    for _, spine in heatmap.spines.items():
        spine.set_visible(True)

    plt.savefig(save_dir+f"mut_{cancer_type}_top20.pdf", dpi=300, bbox_inches = 'tight', pad_inches = 0)


    # plot max-top 5
    top_n = 4
    auc_matrix_reordered = auc_matrix_reordered[["HSC", "mean(ND)", "cv(ND)", "mean(CL)", "mean(HC)"]] 
    if len(auc_matrix_reordered) > top_n:
        max_ass = auc_matrix_reordered.abs().max(axis=1).sort_values(ascending=False)
        max_ass = max_ass[max_ass>0.2]
        max_ass = max_ass.head(top_n).index
        auc_matrix_reordered = auc_matrix_reordered.loc[list(max_ass), :]

    count_ones = Y[auc_matrix_reordered.index].mean(axis=0)

    # Calculate figure size dynamically based on the number of columns (heatmap cells width)
    n_cols = auc_matrix_reordered.shape[0]
    cell_size = 0.25  # size of each cell in inches
    fig_width = n_cols * cell_size
    fig_height = 1.5  # fixed height for consistency

    # Create a barplot above the heatmap
    fig, (ax_bar, ax_heatmap) = plt.subplots(2, 1, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios': [1, 5], 'hspace': 0.05})

    # Barplot
    sns.barplot(x=count_ones.index, y=count_ones.values, ax=ax_bar, color='gray', width=0.5*cell_size)
    ax_bar.set_xticklabels([])
    ax_bar.set_xticks([])
    ax_bar.set_xlabel('')
    ax_bar.set_ylabel('')
    ax_bar.set_title(cancer_type)  # example title, adjust as needed
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_xlim(-0.2, n_cols - 0.2)
    max_y_tick = 0.3
    ax_bar.set_ylim(0, max_y_tick)
    ax_bar.set_yticks([max_y_tick] if ci==0 else [])
    ax_bar.set_yticklabels([f"{max_y_tick}"] if ci==0 else [])

    # Adjust bar width
    for bar in ax_bar.patches:
        bar.set_width(0.8)

    # Heatmap
    sns.heatmap(auc_matrix_reordered.T, cmap="coolwarm", vmin=-1, vmax=1, cbar=False, 
                linewidths=0.5, linecolor='gray', xticklabels=True,
                cbar_kws={'shrink': 0.5, 'label': 'Scaled AUC'}, ax=ax_heatmap)

    for _, spine in ax_heatmap.spines.items():
        spine.set_visible(True)

    if ci != 0:
        plt.yticks([])

    plt.tight_layout()
    fig.savefig(save_dir+f"mut_{cancer_type}_top5.pdf", dpi=300, bbox_inches = 'tight', pad_inches = 0)
    fig.savefig(save_dir+f"mut_{cancer_type}_top5.png", dpi=600, bbox_inches = 'tight', pad_inches = 0)
