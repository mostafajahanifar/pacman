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
from matplotlib.colors import Normalize

def calculate_corr_matrix(df1, df2, method='spearman', pvalue_correction="fdr_bh"):
    if method not in ['spearman', 'pearson']:
        raise ValueError("Method must be 'spearman' or 'pearson'")
    
    # Drop rows with NaN values in Y and align the indices
    non_nan_indices = ~df2.isna().any(axis=1)
    df1 = df1.loc[non_nan_indices]
    df2 = df2.loc[non_nan_indices]

    # scaling the data
    scaler = MinMaxScaler()
    df1 = pd.DataFrame(scaler.fit_transform(df1), columns=df1.columns, index=df1.index)
    df2 = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns, index=df2.index)

    corr_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=np.float32)
    pvalue_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=np.float32)
    for row in df1.columns:
        for col in df2.columns:
            if method == 'spearman':
                corr, pvalue = stats.spearmanr(df1[row], df2[col])
            elif method == 'pearson':
                corr, pvalue = stats.pearsonr(df1[row], df2[col])
            corr_matrix.at[row, col] = np.float32(corr)
            pvalue_matrix.at[row, col] = np.float32(pvalue)
    # correcting pvalues for the number of genes
    if pvalue_correction is not None:
        # Flatten the DataFrame to a 1D array
        pvals = pvalue_matrix.values.flatten()
        # Apply the correction
        corrected_pvals = multipletests(pvals, alpha=0.05, method=pvalue_correction)[1]
        # Reshape the corrected p-values back to the original shape of pvalue_matrix
        corrected_pvals_matrix = corrected_pvals.reshape(pvalue_matrix.shape)
        # Replace the values in the original DataFrame
        pvalue_matrix.loc[:, :] = corrected_pvals_matrix

    return corr_matrix, pvalue_matrix

save_root = "gene/cnv_corr"

# keep only columns that are related to mutations
gene_expr_all = pd.read_csv("gene/data/PORPOISE_data_matched.csv")
sel_cols = [col for col in gene_expr_all.columns if "_cnv" in col]
gene_expr_all = gene_expr_all[['type', 'case_id', 'slide_id'] + sel_cols]

col_rename_dict = {col: col.split("_")[0] for col in sel_cols}
gene_expr_all = gene_expr_all.rename(columns=col_rename_dict)

selected_feats = [
"mit_wsi_count",
"mit_hotspot_count",
"mit_nodeDegrees_mean",
"mit_nodeDegrees_max",
"mit_nodeDegrees_std",
"mit_clusterCoff_mean",
"mit_clusterCoff_std",
"mit_clusterCoff_perc10",
"mit_clusterCoff_perc80",
"mit_cenDegree_mean",
"mit_cenDegree_std",
"mit_cenCloseness_max",
"mit_cenEigen_mean",
"mit_cenEigen_max",
"mit_cenEigen_std",
"mit_cenHarmonic_mean",
"mit_cenHarmonic_std",
]
mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_clinical_merged.csv')
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type"]+selected_feats]
mitosis_feats.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type"] else col for col in mitosis_feats.columns]
mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")

for ci, cancer_type in enumerate(sorted(gene_expr_all["type"].unique())):
    print(f"Working on {cancer_type}")
    save_dir = f"{save_root}/{cancer_type}/"
    os.makedirs(save_dir, exist_ok=True)

    mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"]==cancer_type]
    gene_exp_cancer = gene_expr_all[gene_expr_all["type"]==cancer_type]

    # drop missing mutations
    gene_exp_cancer = gene_exp_cancer.dropna(axis=1, how="all")

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
    Y = df2_common.drop(columns=['case_id', 'type', 'slide_id']).reset_index(drop=True)

    X = X[X.std(axis=0).index[X.std(axis=0)!=0]]

    if len(Y.columns)==0:
        print("No CNV left to process")
        continue


    # Measure feature-mutation association
    corr_matrix, pval_matrix = calculate_corr_matrix(X, Y)

    corr_matrix.to_csv(save_dir+f"cnv_corr-r_{cancer_type}.csv")
    pval_matrix.to_csv(save_dir+f"cnv_corr-p_{cancer_type}.csv")

    auc_matrix = corr_matrix.T
    pval_matrix = pval_matrix.T
    if len(auc_matrix) > 20:
        aucs_sorted = auc_matrix.abs().max(axis=1).sort_values(ascending=False)
        max_ass = aucs_sorted.head(20).index
        auc_matrix = auc_matrix.loc[list(max_ass), :]
        pval_matrix = pval_matrix.loc[list(max_ass), :]
    
    # Perform hierarchical clustering
    row_linkage = linkage(auc_matrix, method='ward')
    col_linkage = linkage(auc_matrix.T, method='ward')

    # Get the order of rows and columns based on clustering
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    # Reorder the data matrix
    auc_matrix_reordered = auc_matrix.iloc[:, col_order].iloc[row_order, :]
    pval_matrix_reordered = pval_matrix.iloc[:, col_order].iloc[row_order, :]

    # Plot the heatmap with reordered data and customization
    plt.figure(figsize=(4, 4))
    annotations = pval_matrix_reordered.applymap(lambda x: '*' if x < 0.05 else '')
    heatmap = sns.heatmap(auc_matrix_reordered, cmap="coolwarm", vmin=-1, vmax=1, cbar=False, 
                        linewidths=0.5, linecolor='gray', square=True,
                        annot=annotations, fmt='', annot_kws={"size": 10, "va": "center_baseline", "ha": "center"},
                        cbar_kws={'shrink': 0.5, 'label': 'Scaled AUC'})


    for _, spine in heatmap.spines.items():
        spine.set_visible(True)

    plt.savefig(save_dir+f"cnv_corr_{cancer_type}_top20.pdf", dpi=300, bbox_inches = 'tight', pad_inches = 0)


    # plot max-top 5
    top_n = 5
    auc_matrix_reordered = auc_matrix_reordered[["HSC", "mean(ND)", "mean(CL)", "mean(DC)", "max(EC)"]] 
    pval_matrix_reordered = pval_matrix_reordered[["HSC", "mean(ND)", "mean(CL)", "mean(DC)", "max(EC)"]] 

    if len(auc_matrix_reordered) > top_n:
        max_ass = auc_matrix_reordered.abs().max(axis=1).sort_values(ascending=False)
        if len(max_ass[max_ass>0.2]) > 5:
            max_ass = max_ass.head(5).index
        elif len(max_ass[max_ass>0.2]) < 3:
            max_ass = max_ass.head(2).index
        else:
            max_ass = max_ass[max_ass>0.2].index    
        auc_matrix_reordered = auc_matrix_reordered.loc[list(max_ass), :]
        pval_matrix_reordered = pval_matrix_reordered.loc[list(max_ass), :]

    # Plotting violin plot and dotpolot 
    n_cols = auc_matrix_reordered.shape[0]
    cell_size = 0.22  # size of each cell in inches
    fig_width = n_cols * cell_size
    fig_height = 2  # adjusted height for the inclusion of violin plot

    # Create a violin plot above the heatmap
    fig, (ax_violin, ax_dotplot) = plt.subplots(2, 1, figsize=(fig_width, fig_height), 
                                                gridspec_kw={'height_ratios': [1, 1.3], 'hspace': 0.1})

    # Prepare data for violin plot
    Y_selected = Y[auc_matrix_reordered.index].melt(var_name='Variable', value_name='CNV')

    # Violin plot
    sns.violinplot(x='Variable', y='CNV', data=Y_selected, ax=ax_violin, inner=None, linewidth=0.1, color="gray")
    ax_violin.set_xticklabels([])
    ax_violin.set_xticks([])
    ax_violin.set_xlabel('')
    ax_violin.set_ylabel('CNV' if ci==0 else '')
    # ax_violin.set_ylabel('')
    ax_violin.set_title(cancer_type)  # example title, adjust as needed
    ax_violin.spines['top'].set_visible(False)
    ax_violin.spines['right'].set_visible(False)
    ax_violin.spines['left'].set_visible(True)
    ax_violin.spines['bottom'].set_visible(True)
    ax_violin.set_ylim(-2.5, 2.5)
    ax_violin.set_yticks([-2, -1, 0, 1, 2])
    ax_violin.set_yticklabels([-2, -1, 0, 1, 2] if ci==0 else [])

    # Heatmap
    annotations = pval_matrix_reordered.applymap(lambda x: '*' if x < 0.05 else '')

    sns.heatmap(auc_matrix_reordered.T, cmap="coolwarm", vmin=-1, vmax=1, cbar=False, 
                linewidths=0.5, linecolor='gray', xticklabels=True, yticklabels=True,
                annot=annotations.T, fmt='', annot_kws={"size": 10, "va": "center_baseline", "ha": "center"},
                cbar_kws={'shrink': 0.5, 'label': 'Scaled AUC'}, ax=ax_dotplot)

    for _, spine in ax_dotplot.spines.items():
        spine.set_visible(True)

    if ci != 0:
        plt.yticks([])


    plt.tight_layout()
    fig.savefig(save_dir+f"cnv_corr_{cancer_type}_top5.pdf", dpi=300, bbox_inches = 'tight', pad_inches = 0)
    fig.savefig(save_dir+f"cnv_corr_{cancer_type}_top5.png", dpi=600, bbox_inches = 'tight', pad_inches = 0)
