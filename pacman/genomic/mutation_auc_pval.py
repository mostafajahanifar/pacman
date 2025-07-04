import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR

print(7 * "=" * 7)
print("Measuring association with point mutation")
print(7 * "=" * 7)


def auc_association_matrix(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with NaN values in Y and align the indices
    non_nan_indices = ~Y.isna().any(axis=1)
    X_clean = X  # .loc[non_nan_indices]
    Y_clean = Y  # .loc[non_nan_indices]

    # Standardize the columns of X to the 0-1 range
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index
    )

    # Initialize an empty DataFrame for the AUC association matrix
    auc_matrix = pd.DataFrame(index=X.columns, columns=Y.columns)

    # Iterate over each combination of columns from X and Y
    for x_col in X.columns:
        for y_col in Y.columns:
            df_no_na = pd.concat([X_scaled[x_col], Y_clean[y_col]], axis=1)
            df_no_na = df_no_na.dropna(axis=0, how="any")
            try:
                # Compute the AUC score for the current pair of columns
                auc_score = roc_auc_score(df_no_na[y_col], df_no_na[x_col])
            except Exception as e:
                print(e, y_col, df_no_na[y_col].unique())
                auc_score = 0.5
            # Store the AUC score in the matrix
            auc_matrix.at[x_col, y_col] = 2 * (auc_score - 0.5)

    return auc_matrix.astype(float).T


def permutation_test(X, Y, auc_matrix, n_permutations=500):
    n_rows, n_cols = Y.shape
    auc_matrix_perm_all = []

    for i in tqdm(
        range(n_permutations), total=n_permutations, ascii=True, desc="Permutation"
    ):
        Y_perm = Y.apply(np.random.permutation)  # Permute each column
        auc_matrix_perm = auc_association_matrix(X, Y_perm)
        auc_matrix_perm_all.append(auc_matrix_perm)

    auc_matrix_perm_all = np.array(auc_matrix_perm_all)
    p_values = pd.DataFrame(index=auc_matrix.index, columns=auc_matrix.columns)

    for x_col in auc_matrix.columns:
        for y_col in auc_matrix.index:
            original_auc = auc_matrix.loc[y_col, x_col]
            permuted_aucs = auc_matrix_perm_all[
                :, auc_matrix.index.get_loc(y_col), auc_matrix.columns.get_loc(x_col)
            ]
            p_val = np.mean(np.abs(permuted_aucs) > np.abs(original_auc))
            p_values.at[y_col, x_col] = p_val

    return p_values


save_root = f"{RESULTS_DIR}/genomic/mutation_pval"

selected_feats = [
    "HSC",
    "mean(ND)",
    "cv(ND)",
    "mean(CL)",
    "mean(HC)",
]

# reading necessary data
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type"] + selected_feats]
gene_expr_all = pd.read_csv(f"{DATA_DIR}/tcga_all_gene_mutations.csv")

# drop missing mutations
gene_exp_cancer = gene_expr_all.dropna(axis=1, how="all")

for ci, cancer_type in enumerate(sorted(gene_expr_all["type"].unique())):

    print(f"Working on {cancer_type}")
    save_dir = f"{save_root}/{cancer_type}/"
    os.makedirs(save_dir, exist_ok=True)

    mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"] == cancer_type]
    # gene_exp_cancer = gene_expr_all[gene_expr_all["type"]==cancer_type]

    # Find the common case names between mitosis features and gene expressions
    common_cases = pd.Series(
        list(
            set(mitosis_feats_cancer["bcr_patient_barcode"]).intersection(
                set(gene_exp_cancer["case_id"])
            )
        )
    )
    ## Keep only the rows with the common case names in both dataframes
    df1_common = mitosis_feats_cancer[
        mitosis_feats_cancer["bcr_patient_barcode"].isin(common_cases)
    ]
    df2_common = gene_exp_cancer[gene_exp_cancer["case_id"].isin(common_cases)]
    df2_common = df2_common.drop_duplicates(subset="case_id")

    ## Sort the dataframes based on 'case_name'
    df1_common = df1_common.sort_values("bcr_patient_barcode")
    df2_common = df2_common.sort_values("case_id")

    X = df1_common.drop(columns=["bcr_patient_barcode", "type"]).reset_index(drop=True)
    Y = df2_common.drop(columns=["case_id", "type"]).reset_index(drop=True)

    # drop genes with less than 4% mutations
    mut_thresh = int(0.04 * len(Y))
    Y = Y[Y.sum(axis=0).index[Y.sum(axis=0) > mut_thresh]]

    # drop mutations with zero std (only one label)
    Y = Y[Y.std(axis=0).index[Y.std(axis=0) != 0]]

    # remove duplicated mutations
    Y = Y.loc[:, ~Y.columns.duplicated()].copy()

    # Measure feature-mutation association
    auc_matrix = auc_association_matrix(X, Y)
    auc_matrix.to_csv(save_dir + f"{cancer_type}_auc_matrix.csv")

    pval_matrix = permutation_test(X, Y, auc_matrix, n_permutations=500)
    pval_matrix.to_csv(save_dir + f"{cancer_type}_pval_matrix.csv")

    # Plot the top 20
    if len(auc_matrix) > 20:
        aucs_sorted = auc_matrix.abs().max(axis=1).sort_values(ascending=False)
        max_ass = aucs_sorted.head(20).index
        auc_matrix = auc_matrix.loc[list(max_ass), :]
        pval_matrix = pval_matrix.loc[list(max_ass), :]

    # Perform hierarchical clustering
    row_linkage = linkage(auc_matrix, method="ward")
    col_linkage = linkage(auc_matrix.T, method="ward")

    # Get the order of rows and columns based on clustering
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    # Reorder the data matrix
    auc_matrix_reordered = auc_matrix.iloc[:, col_order].iloc[row_order, :]
    pval_matrix_reordered = pval_matrix.iloc[:, col_order].iloc[row_order, :]
    annot_matrix_reordered = pval_matrix_reordered.applymap(
        lambda x: "*" if x <= 0.05 else ""
    )
    # Plot the heatmap with reordered data and customization
    plt.figure(figsize=(4, 4))
    heatmap = sns.heatmap(
        auc_matrix_reordered,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        square=True,
        annot=annot_matrix_reordered,
        fmt="",
        annot_kws={"size": 10, "va": "center_baseline", "ha": "center"},
        cbar_kws={"shrink": 0.5, "label": r"Scaled AUC's $\rho$"},
    )

    for _, spine in heatmap.spines.items():
        spine.set_visible(True)

    plt.savefig(
        save_dir + f"mut_{cancer_type}_top20.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    # plot max-top 5
    top_n = 4
    auc_matrix_reordered = auc_matrix_reordered[
        ["HSC", "mean(ND)", "cv(ND)", "mean(CL)", "mean(HC)"]
    ]
    pval_matrix_reordered = pval_matrix_reordered[
        ["HSC", "mean(ND)", "cv(ND)", "mean(CL)", "mean(HC)"]
    ]
    if len(auc_matrix_reordered) > top_n:
        max_ass = auc_matrix_reordered.abs().max(axis=1).sort_values(ascending=False)
        max_ass = max_ass[max_ass > 0.2]
        max_ass = max_ass.head(top_n).index
        auc_matrix_reordered = auc_matrix_reordered.loc[list(max_ass), :]
        pval_matrix_reordered = pval_matrix_reordered.loc[list(max_ass), :]

    count_ones = Y[auc_matrix_reordered.index].mean(axis=0)

    # Calculate figure size dynamically based on the number of columns (heatmap cells width)
    n_cols = auc_matrix_reordered.shape[0]
    cell_size = 0.25  # size of each cell in inches
    fig_width = n_cols * cell_size
    fig_height = 1.5  # fixed height for consistency

    # Create a barplot above the heatmap
    fig, (ax_bar, ax_heatmap) = plt.subplots(
        2,
        1,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0.05},
    )

    # Barplot
    sns.barplot(x=count_ones.index, y=count_ones.values, ax=ax_bar, color="gray")  #
    # Adjust bar width
    for bar in ax_bar.patches:
        bar.set_width(0.8)

    ax_bar.set_xticklabels([])
    ax_bar.set_xticks([])
    ax_bar.set_xlabel("")
    ax_bar.set_ylabel("")
    ax_bar.set_title(cancer_type)  # example title, adjust as needed
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    # ax_bar.set_xlim(-0.5, n_cols-0.6)
    max_y_tick = 0.3
    ax_bar.set_ylim(0, max_y_tick)
    ax_bar.set_yticks([max_y_tick] if ci == 0 else [])
    ax_bar.set_yticklabels([f"{max_y_tick}"] if ci == 0 else [])

    # Heatmap
    annot_matrix_reordered = pval_matrix_reordered.applymap(
        lambda x: "*" if x <= 0.05 else ""
    )
    heatmap = sns.heatmap(
        auc_matrix_reordered.T,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        square=True,
        annot=annot_matrix_reordered.T,
        fmt="",
        annot_kws={"size": 10, "va": "center_baseline", "ha": "center"},
        cbar_kws={"shrink": 0.5, "label": r"Scaled AUC's $\rho$"},
    )

    for _, spine in ax_heatmap.spines.items():
        spine.set_visible(True)

    if ci != 0:
        plt.yticks([])

    plt.tight_layout()
    # fig.savefig(save_dir+f"mut_{cancer_type}_top5.pdf", dpi=300, bbox_inches = 'tight', pad_inches = 0.01, transparent=True)
    fig.savefig(
        save_dir + f"mut_{cancer_type}_top5.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.01,
        transparent=True,
    )
