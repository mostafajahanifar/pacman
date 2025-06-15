import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines.utils import concordance_index
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from pacman.config import DATA_DIR, RESULTS_DIR

print(7 * "=" * 7)
print("Comparing C-indices of different features for survival ranking")
print(7 * "=" * 7)

PERM_N = 1000 # Number of permutations for significance testing

def adjust_p_values(pval_df, alpha=0.05, method="fdr_bh"):
    """
    Adjusts p-values for multiple testing using the specified method.

    Args:
        pval_df (pd.DataFrame): DataFrame of raw p-values.
        alpha (float): Significance level.
        method (str): Method for p-value correction. 'fdr_bh' for Benjamini-Hochberg.

    Returns:
        pd.DataFrame: Boolean DataFrame indicating significance after correction.
    """
    pvals = pval_df.values.flatten()
    mask = ~pd.isnull(pvals)

    corrected = multipletests(pvals[mask], alpha=alpha, method=method)

    significant = np.full(pvals.shape, False)
    significant[mask] = corrected[0]  # corrected[0] is the reject list

    sig_matrix = pd.DataFrame(significant.reshape(pval_df.shape), 
                              index=pval_df.index, 
                              columns=pval_df.columns)
    return sig_matrix


def bootstrap_c_index(group, feature, time_col, event_col, n_bootstrap=1000, alpha=0.05):
    """
    Bootstrap-based significance test to check if c-index is significantly different from 0.5.
    
    Returns:
        mean_cindex, p_value
    """
    times = group[time_col].values
    events = group[event_col].values
    scores = -group[feature].values  # assuming higher = riskier
    EE = group[event_col].to_numpy()
    rng = np.random.RandomState()

    n = len(group)
    boot_cis = []

    for _ in range(n_bootstrap):
        # idx = np.random.choice(n, n, replace=True)
        idx = list(rng.choice(np.nonzero(EE==0)[0],size = len(EE)-np.sum(EE),replace=True))+list(rng.choice(np.nonzero(EE==1)[0],size = np.sum(EE),replace=True))
        try:
            ci = concordance_index(times[idx], scores[idx], events[idx])
            boot_cis.append(ci)
        except:
            continue

    boot_cis = np.array(boot_cis)
    mean_ci = np.mean(boot_cis)

    # Two-sided non-parametric test
    p_value = 2 * min(
        np.mean(boot_cis <= 0.5),
        np.mean(boot_cis >= 0.5)
    )
    return mean_ci, p_value

def permutation_c_index_test(group, feature, time_col, event_col, n_permutations=1000):
    times = group[time_col].values
    events = group[event_col].values
    scores = -group[feature].values  # assume higher = riskier

    obs_ci = concordance_index(times, scores, events)
    null_cis = []
    rng = np.random.RandomState()

    for _ in tqdm(range(n_permutations), total=n_permutations, desc=f"{feature}"):
        shuffled_scores = rng.permutation(scores)
        try:
            null_ci = concordance_index(times, shuffled_scores, events)
            null_cis.append(null_ci)
        except:
            continue

    null_cis = np.array(null_cis)
    p_value = np.mean(np.abs(null_cis - 0.5) >= np.abs(obs_ci - 0.5))
    return obs_ci, p_value


def plot_cindex_heatmap_with_significance(df, sig_matrix, figsize=(10, 6), cmap="viridis"):
    """
    Plots a heatmap of C-index values with * for statistically significant cells.

    Args:
        df (pd.DataFrame): C-index values
        sig_matrix (pd.DataFrame): Boolean matrix (same shape) where True = significant

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    annot = df.round(2).astype(str)
    annot = annot.mask(~sig_matrix, annot)  # keep only significant cells
    annot = annot.where(~sig_matrix, annot + "*")  # add asterisk

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=annot, fmt="", cmap=cmap, linewidths=0.5, linecolor='gray',
                vmin=0.2, vmax=0.8, ax=ax)
    ax.set_ylabel("Cancer Type")
    ax.set_xlabel("Feature")
    # fig.tight_layout()
    return fig, ax


def compute_cindex_and_significance(df, features, time_col, event_col, type_col, alpha=0.05):
    cindex_data = {}
    significance = {}

    for cancer_type, group in df.groupby(type_col):
        print(f"Processing {cancer_type}...")
        cindex_row = {}
        sig_row = {}
        for feature in features:
            try:
                # mean_ci, p_val = bootstrap_c_index(group, feature, time_col, event_col)
                mean_ci, p_val = permutation_c_index_test(group, feature, time_col, event_col, n_permutations=PERM_N)
                cindex_row[feature] = mean_ci
                sig_row[feature] = p_val
            except:
                cindex_row[feature] = np.nan
                sig_row[feature] = False
        cindex_data[cancer_type] = cindex_row
        significance[cancer_type] = sig_row

    cindex_df = pd.DataFrame(cindex_data).T
    sig_df = pd.DataFrame(significance).T
    sig_df = adjust_p_values(sig_df, alpha=alpha, method="fdr_bh")
    return cindex_df, sig_df

save_dir = f"{RESULTS_DIR}/survival/cindex/"
os.makedirs(save_dir, exist_ok=True)

features = ["HSC", "mean(ND)", "cv(ND)","mean(CL)","mean(HC)", "AMH", "AMAH", "AFW"] # "cv(ND)","per99(ND)", ,"std(CL)","per99(CL)","std(HC)","per10(HC)"
censor_at = 120

for event_col in ["DSS", "PFI", "OS", "DFI"]:
    time_col = f'{event_col}.time'


    mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))

    mitosis_feats = mitosis_feats.dropna(subset=[event_col, time_col])
    mitosis_feats[event_col] = mitosis_feats[event_col].astype(int)
    mitosis_feats[time_col] = (mitosis_feats[time_col]/30.4).astype(int)

    if event_col=="PFI":
        valid_types = ["BLCA", "BRCA", "CESC", "COADREAD", "ESCA", "GBMLGG", "HNSC", "KICH", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "OV", "PAAD", "SKCM", "STAD", "UCEC", "THCA", "PRAD", "SARC", "TGCT", "ACC", "MESO", "THYM", "CHOL"]
    elif event_col=="DFI":
        valid_types = ["BLCA", "BRCA", "CESC", "COADREAD", "ESCA", "GBMLGG", "HNSC", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "OV", "PAAD", "SARC", "STAD", "TGCT", "UCEC"]
    elif event_col=="OS":
        valid_types = ["BLCA", "BRCA", "CESC", "COADREAD", "ESCA", "GBMLGG", "HNSC", "KICH", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "OV", "PAAD", "SKCM", "STAD", "UCEC", "SARC", "ACC", "MESO", "CHOL"]
    elif event_col=="DSS":
        valid_types = ["BLCA", "BRCA", "CESC", "COADREAD", "ESCA", "GBMLGG", "HNSC", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "OV", "PAAD", "SKCM", "STAD", "UCEC", "SARC", "ACC", "MESO", "CHOL"]

    mitosis_feats = mitosis_feats[mitosis_feats["type"].isin(valid_types)]

    mitosis_feats= mitosis_feats.sort_values(by="type", ascending=True)

    if censor_at > 0:
        mitosis_feats.loc[mitosis_feats[time_col] > censor_at, event_col] = 0
        mitosis_feats.loc[mitosis_feats[time_col] > censor_at, time_col] = censor_at


    c_index_table, significance_table = compute_cindex_and_significance(
        mitosis_feats, features, time_col, event_col, "type"
    )

    # # Sort columns from highest to lowest average absolute deviation from 0.5
    # col_order = (c_index_table - 0.5).abs().mean().sort_values(ascending=False).index.tolist()
    # c_index_table = c_index_table[col_order]
    # significance_table = significance_table[col_order]


    num_cancer = len(mitosis_feats["type"].unique())
    fig_size = (7, 0.24*num_cancer + 1)
    print(f"Figure size: {fig_size}")
    fig, ax = plot_cindex_heatmap_with_significance(c_index_table, significance_table, figsize=fig_size, cmap="coolwarm")
    ax.set_title(f"C-index for {event_col} (* : permutation test's p-value<0.05)")
    fig.savefig(save_dir+f"cindex_heatmap_{event_col}_censor{censor_at}.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
    c_index_table.to_csv(save_dir+f"cindex_table_{event_col}_censor{censor_at}.csv", index=True)
    significance_table.to_csv(save_dir+f"pvalue_table_{event_col}_censor{censor_at}.csv", index=True)