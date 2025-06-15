import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR
from pacman.utils import calculate_corr_matrix

print(7*"="*7)
print("Measuring correlation of mitotic activity and CIN signatures")
print(7*"="*7)

# Setting CIN signiture nnumbers. the exact name of them can be found onlie, doi: 10.1038/s41586-022-04789-9
# cin_sigs = [f"CX{i}" for i in range (1,18)]
cin_sigs = [f"CX{i}" for i in [1, 2, 3, 4, 5, 6, 14]]

# keep only columns that are related to mutations
cin_df = pd.read_excel(os.path.join(DATA_DIR,"tcga_cin_signatures.xlsx"), sheet_name="ST_18_TCGA_Activities_raw")
cin_df = cin_df.rename(columns={"Unnamed: 0": "TCGA Participant Barcode"})

atyp_feat = "AMAH" # the atypical mitotic activit measure
selected_feats = [
    "mean(ND)",
    "AMAH",
    "AFW",
]

#reading necessary data
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]+selected_feats]

# creating save directory
save_root = os.path.join(RESULTS_DIR, "morphology", "cin_correlation")
os.makedirs(save_root, exist_ok=True)


all_corr = []
all_pval = []

for cancer_type in ALL_CANCERS + ["Mitotic Hot", "Mitotic Cold", "Pan-cancer"]:

    if cancer_type in ["Mitotic Hot", "Mitotic Cold", "Pan-cancer"]:
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"].isin(ALL_CANCERS)]
    else:
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"]==cancer_type]

    if cancer_type=="Mitotic Hot":
        mitosis_feats_cancer = mitosis_feats_cancer[mitosis_feats_cancer["temperature"]=="Hot"]
    if cancer_type=="Mitotic Cold":
        mitosis_feats_cancer = mitosis_feats_cancer[mitosis_feats_cancer["temperature"]=="Cold"]

    gene_exp_cancer = cin_df.copy()


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

    if len(df2_common) < 50:
        continue

    X = df1_common[[atyp_feat]]
    Y = df2_common[cin_sigs]
    corr_matrix, pvalue_matrix = calculate_corr_matrix(Y, X, method='spearman', pvalue_correction="fdr_bh")

    all_corr.append(corr_matrix.rename(columns={atyp_feat: cancer_type}))
    all_pval.append(pvalue_matrix.rename(columns={atyp_feat: cancer_type}))

corr_matrix = pd.concat(all_corr, axis=1)
pvalue_matrix = pd.concat(all_pval, axis=1)
annotations = pvalue_matrix.applymap(lambda x: '*' if x < 0.05 else '') 

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 4.4), gridspec_kw={'width_ratios': [corr_matrix.shape[1]-3, 2, 1]})
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


plt.savefig(f"{save_root}/cin_correlations_{atyp_feat}_selected_spearmanFDR.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
corr_matrix.to_csv(f"{save_root}/cin_correlations_{atyp_feat}_selected_spearmanFDR_corr.csv")
pvalue_matrix.to_csv(f"{save_root}/cin_correlations_{atyp_feat}_selected_spearmanFDR_pval.csv")
