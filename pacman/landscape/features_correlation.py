import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR


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

ALL_CANCERS = sorted(['SARC',
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
 ])


immune_feats = [
    "Proliferation"]

# keep only columns that are related to mutations
immune_df = pd.read_excel("gene/data/tcga_all_immune.xlsx")
immune_df["TCGA Study"] = immune_df["TCGA Study"].replace(["COAD", "READ"], "COADREAD")
immune_df["TCGA Study"] = immune_df["TCGA Study"].replace(["GBM", "LGG"], "GBMLGG")

selected_feats = [
    "HSC",
    "mean(ND)",
    # "cv(ND)",
    "mean(CL)",
    # "mit_clusterCoff_std",
    "mean(HC)",
    # "mit_cenHarmonic_per10",
]

mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancerNew_withAtypicalNew.csv')
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]+selected_feats]

mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")

all_corr = []
all_pval = []

for cancer_type in ALL_CANCERS + ["Mitotic Hot", "Mitotic Cold", "Pan-cancer"]:

    if cancer_type in ["Mitotic Hot", "Mitotic Cold", "Pan-cancer"]:
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"].isin(ALL_CANCERS)]
        gene_exp_cancer = immune_df[immune_df["TCGA Study"].isin(ALL_CANCERS)]
    else:
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"]==cancer_type]
        gene_exp_cancer = immune_df[immune_df["TCGA Study"]==cancer_type]

    if cancer_type=="Mitotic Hot":
        mitosis_feats_cancer = mitosis_feats_cancer[mitosis_feats_cancer["temperature"]=="Hot"]
    if cancer_type=="Mitotic Cold":
        mitosis_feats_cancer = mitosis_feats_cancer[mitosis_feats_cancer["temperature"]=="Cold"]


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

    Y = df1_common[selected_feats]
    Y.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type", "temperature"] else col for col in Y.columns]
    X = df2_common[immune_feats]
    corr_matrix, pvalue_matrix = calculate_corr_matrix(Y, X, method='pearson', pvalue_correction="bonferroni")

    all_corr.append(corr_matrix.rename(columns={"Proliferation": cancer_type}))
    all_pval.append(pvalue_matrix.rename(columns={"Proliferation": cancer_type}))

corr_matrix = pd.concat(all_corr, axis=1)
pvalue_matrix = pd.concat(all_pval, axis=1)
annotations = pvalue_matrix.applymap(lambda x: '*' if x < 0.05 else '') 

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 7.2), gridspec_kw={'width_ratios': [corr_matrix.shape[1]-3, 2, 1]})
# Plot the heatmap for all but the last column
heatmap1 = sns.heatmap(corr_matrix.iloc[:, :-3], cmap="coolwarm", vmin=-1, vmax=1, cbar=False, 
                       linewidths=0.5, linecolor='gray', square=True,
                       annot=annotations.iloc[:, :-3], fmt='', annot_kws={"size": 8, "va": "center_baseline", "ha": "center"},
                       yticklabels=True, xticklabels=True, ax=ax1)

heatmap2 = sns.heatmap(corr_matrix.iloc[:, -3:-1], cmap="coolwarm", vmin=-1, vmax=1, cbar=False, 
                       linewidths=0.5, linecolor='gray', square=True,
                       annot=annotations.iloc[:, -3:-1], fmt='', annot_kws={"size": 8, "va": "center_baseline", "ha": "center"},
                       yticklabels=False, xticklabels=True, ax=ax2)

# Plot the heatmap for the last column
heatmap3 = sns.heatmap(corr_matrix.iloc[:, -1:], cmap="coolwarm", vmin=-1, vmax=1, cbar=False, 
                       linewidths=0.5, linecolor='gray', square=True,
                       annot=annotations.iloc[:, -1:], fmt='', annot_kws={"size": 8, "va": "center_baseline", "ha": "center"},
                       yticklabels=False, xticklabels=True, ax=ax3)

for _, spine in heatmap1.spines.items():
    spine.set_visible(True)

for _, spine in heatmap2.spines.items():
    spine.set_visible(True)

for _, spine in heatmap3.spines.items():
    spine.set_visible(True)

for label in heatmap2.get_xticklabels():
    label.set_rotation(90)

for label in heatmap3.get_xticklabels():
    label.set_rotation(90)
plt.subplots_adjust(wspace=0.03, hspace=0)

plt.savefig("results_final/landscape/proliferation_correlations.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
plt.savefig("results_final/landscape/proliferation_correlations.pdf", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)