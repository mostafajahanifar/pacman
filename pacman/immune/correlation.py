import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR
from pacman.utils import calculate_corr_matrix

print(7 * "=" * 7)
print("Running Correlation Analysis between Mitosis Features and Immune Features")
print(7 * "=" * 7)

# Define the immune features to be used in the analysis
immune_feats = [
                # selected immune features
                "Th1 Cells","Th2 Cells","Th17 Cells",
                "NK Cells Activated","NK Cells Resting",
                "T Cells Regulatory Tregs",
                "Monocytes","Macrophages M0","Macrophages M1","Macrophages M2",

                ## Other possible immune features
                # "Proliferation",
                #  "Th1:Th2 cells ratio",
                # "Dendritic Cells Activated","Dendritic Cells Resting",
                # "Mast Cells Activated","Mast Cells Resting",
                # "Plasma Cells",
                # "T Cells CD4 Memory Activated","T Cells CD4 Memory Resting","T Cells CD4 Naive",
                # "T Cells CD8",
                # "T Cells Follicular Helper",
                # "Lymphocytes","Neutrophils","Eosinophils","Mast Cells","Dendritic Cells","Macrophages"
                ]

selected_feats = [
"mean(ND)",
"AFW",
"AMAH",
]
feat = "mean(ND)" # the feature for which collaboration is done

#reading necessary data
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]+selected_feats]

immune_df = pd.read_csv(os.path.join(DATA_DIR, "tcga_all_immune_new.csv"))
# drop missing data
immune_df = immune_df.dropna(axis=1, how="all")


all_corr = []
all_pval = []

for cancer_type in ALL_CANCERS + ["Mitotic Hot", "Mitotic Cold", "All"]:

    if cancer_type in ["Mitotic Hot", "Mitotic Cold", "All"]:
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"].isin(ALL_CANCERS)]
    else:
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"]==cancer_type]

    if cancer_type=="Mitotic Hot":
        mitosis_feats_cancer = mitosis_feats_cancer[mitosis_feats_cancer["temperature"]=="Hot"]
    if cancer_type=="Mitotic Cold":
        mitosis_feats_cancer = mitosis_feats_cancer[mitosis_feats_cancer["temperature"]=="Cold"]

    # Find the common case names between mitosis features and gene expressions
    common_cases = pd.Series(list(set(mitosis_feats_cancer['bcr_patient_barcode']).intersection(set(immune_df['TCGA Participant Barcode']))))
    ## Keep only the rows with the common case names in both dataframes
    df1_common = mitosis_feats_cancer[mitosis_feats_cancer['bcr_patient_barcode'].isin(common_cases)]
    df2_common = immune_df[immune_df['TCGA Participant Barcode'].isin(common_cases)]
    df2_common = df2_common.drop_duplicates(subset='TCGA Participant Barcode')

    ## Sort the dataframes based on 'case_name'
    df1_common = df1_common.sort_values('bcr_patient_barcode')
    df2_common = df2_common.sort_values('TCGA Participant Barcode')

    df1_common = df1_common.reset_index(drop=True)
    df2_common = df2_common.reset_index(drop=True)

    X = df1_common[[feat]]
    Y = df2_common[immune_feats]
    corr_matrix, pvalue_matrix = calculate_corr_matrix(Y, X, method='spearman', pvalue_correction="bonferroni")

    all_corr.append(corr_matrix.rename(columns={feat: cancer_type}))
    all_pval.append(pvalue_matrix.rename(columns={feat: cancer_type}))

corr_matrix = pd.concat(all_corr, axis=1)
pvalue_matrix = pd.concat(all_pval, axis=1)
save_dir = os.path.join(RESULTS_DIR, "immune")
os.makedirs(save_dir, exist_ok=True)
corr_matrix.to_csv(f"{save_dir}/{feat}_correlations_r.csv")
pvalue_matrix.to_csv(f"{save_dir}/{feat}_correlations_p.csv")

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

plt.savefig(f"{save_dir}/{feat}_correlations_new.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)