from itertools import cycle
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def calculate_corr_matrix(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    method: str = "spearman",
    pvalue_correction: Optional[str] = "fdr_bh"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise correlation and p-value matrices between the columns of two dataframes.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe (e.g., features or gene expressions).
    df2 : pd.DataFrame
        Second dataframe (e.g., phenotypes or clinical variables).
    method : str, optional
        Correlation method to use: "spearman" (default) or "pearson".
    pvalue_correction : str or None, optional
        Method for multiple testing correction on p-values (e.g., "fdr_bh", "bonferroni").
        If None, raw p-values are returned.

    Returns
    -------
    corr_matrix : pd.DataFrame
        Matrix of correlation coefficients (shape: df1.columns x df2.columns).
    pvalue_matrix : pd.DataFrame
        Matrix of raw or adjusted p-values (shape: df1.columns x df2.columns).

    Raises
    ------
    ValueError
        If an unsupported correlation method is specified.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError("Method must be 'spearman' or 'pearson'")

    corr_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=np.float32)
    pvalue_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=np.float32)

    for row in df1.columns:
        for col in df2.columns:
            x = df1[row]
            y = df2[col]
            if method == "spearman":
                corr, pval = stats.spearmanr(x, y)
            else:
                corr, pval = stats.pearsonr(x, y)

            corr_matrix.at[row, col] = np.float32(corr)
            pvalue_matrix.at[row, col] = np.float32(pval)

    if pvalue_correction is not None:
        flat_pvals = pvalue_matrix.values.flatten()
        corrected_pvals = multipletests(flat_pvals, alpha=0.05, method=pvalue_correction)[1]
        pvalue_matrix.loc[:, :] = corrected_pvals.reshape(pvalue_matrix.shape)

    return corr_matrix, pvalue_matrix


def get_colors_dict():
    domain_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COADREAD', 'DLBC', 'ESCA', 'GBMLGG', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']
    # Use Set3 color palette from Matplotlib
    set3_palette = list(plt.cm.tab20.colors) + [plt.cm.tab20b.colors[i] for i in [0, 2, 4, 5, 8, 9, 13, 16]] + [plt.cm.tab20c.colors[i] for i in [4, 16]] # plt.cm.tab20.colors + plt.cm.tab20b.colors
    
    # Create a cycle iterator for the colors
    color_cycle = cycle(set3_palette)

    # Generate custom color dictionary
    custom_colors = {domain: next(color_cycle) for domain in domain_list}

    return custom_colors