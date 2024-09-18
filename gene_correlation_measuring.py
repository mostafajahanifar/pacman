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


def calculate_corr_matrix(df1, df2, method='pearson', pvalue_correction="fdr_bh"):
    if method not in ['spearman', 'pearson']:
        raise ValueError("Method must be 'spearman' or 'pearson'")
    
    corr_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=np.float32)
    pvalue_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=np.float32)
    for row in df1.columns:
        for col in df2.columns:
            df_no_na = pd.concat([df1[row], df2[col]], axis=1)
            df_no_na = df_no_na.dropna(axis=0, how="any")
            if method == 'spearman':
                corr, pvalue = stats.spearmanr(df_no_na[row], df_no_na[col])
            elif method == 'pearson':
                corr, pvalue = stats.pearsonr(df_no_na[row], df_no_na[col])
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

selected_feats = [
    "mit_hotspot_count",
    "mit_nodeDegrees_mean",
    "mit_nodeDegrees_cv",
    # "mit_nodeDegrees_per99",
    "mit_clusterCoff_mean",
    # "mit_clusterCoff_std",
    # "mit_clusterCoff_per90",
    "mit_cenHarmonic_mean",
    # "mit_cenHarmonic_std",
    # "mit_cenHarmonic_per99",
    # "mit_cenHarmonic_per10",
]

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Gene to mitosis features analysis')
    # parser.add_argument('--cancer_types', nargs='+', required=True)
    # args = parser.parse_args()
    # cancer_types = args.cancer_types

    #reading necessary data
    mitosis_feats = pd.read_csv('/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_final.csv')
    # mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final.csv')
    # mitosis_feats = pd.read_csv('D:/tcga/tcga_mitosis_ClusterByCancer.csv')
    mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
    mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")
    mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type"]+selected_feats]
    mitosis_feats.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type"] else col for col in mitosis_feats.columns]

    gene_expr_all = pd.read_csv("gene/data/tcga_all_gene_methylation.csv")
    # gene_expr_all["type"] = gene_expr_all["type"].replace(["COAD", "READ"], "COADREAD")
    # gene_expr_all["type"] = gene_expr_all["type"].replace(["GBM", "LGG"], "GBMLGG")

    for cancer_types in ["all"]+ALL_CANCERS:
        print(cancer_types)
        cancer_types = [cancer_types]
        if cancer_types != ["all"]:
            print(f"Working on cancer types: {cancer_types}")
            mitosis_feats_cancer = mitosis_feats.loc[mitosis_feats['type'].isin(cancer_types)] # cancer types should be given in input argparse
        else:
            mitosis_feats_cancer = mitosis_feats.loc[mitosis_feats['type'].isin(ALL_CANCERS)]
            print("Working on ALL cancer types together")

        cancer_types_name = ''.join(cancer_types).upper()
        save_root = f"results_final/gene/methylation/{cancer_types_name}/"
        os.makedirs(save_root, exist_ok=True)

        if cancer_types != ["all"]:
            gene_expr = gene_expr_all[gene_expr_all["type"]==cancer_types_name]
        else:
            gene_expr = gene_expr_all.loc[gene_expr_all['type'].isin(ALL_CANCERS)]

        # Find the common case names between mitosis features and gene expressions
        common_cases = pd.Series(list(set(mitosis_feats_cancer['bcr_patient_barcode']).intersection(set(gene_expr['case_id']))))
        ## Keep only the rows with the common case names in both dataframes
        df1_common = mitosis_feats_cancer[mitosis_feats_cancer['bcr_patient_barcode'].isin(common_cases)]
        df2_common = gene_expr[gene_expr['case_id'].isin(common_cases)]
        ## Sort the dataframes based on 'case_name'
        df1_common = df1_common.sort_values('bcr_patient_barcode')
        df2_common = df2_common.sort_values('case_id')
        ## Remove duplicate rows based on 'case_name' in df2_common
        df2_common = df2_common.drop_duplicates(subset='case_id')
        ## keep only feature and gene data
        X = df1_common.drop(columns=["bcr_patient_barcode", "type"])#.values
        Y = df2_common.drop(columns=['case_id', "type"])
       
        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        if len(X)<10:
            print(f"Only {len(X)} pairs found, not enough to correlate...skipped")
            continue
        print(f"Correlating {len(X)} pairs...")

        corr_matrix, pvalue_matrix = calculate_corr_matrix(X, Y, method='spearman', pvalue_correction="fdr_bh")

        corr_matrix.to_csv(save_root+"corr_r.csv")
        pvalue_matrix.to_csv(save_root+"corr_p.csv")