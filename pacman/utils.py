from itertools import cycle
from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.plotting import (is_latex_enabled, move_spines, remove_spines,
                                remove_ticks)
from matplotlib import pyplot as plt
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


def get_colors_dict():
    domain_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COADREAD', 'DLBC', 'ESCA', 'GBMLGG', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']
    # Use Set3 color palette from Matplotlib
    set3_palette = list(plt.cm.tab20.colors) + [plt.cm.tab20b.colors[i] for i in [0, 2, 4, 5, 8, 9, 13, 16]] + [plt.cm.tab20c.colors[i] for i in [4, 16]] # plt.cm.tab20.colors + plt.cm.tab20b.colors
    
    # Create a cycle iterator for the colors
    color_cycle = cycle(set3_palette)

    # Generate custom color dictionary
    custom_colors = {domain: next(color_cycle) for domain in domain_list}

    return custom_colors

def add_at_risk_counts(
    *fitters,
    labels: Optional[Union[Iterable, bool]] = None,
    rows_to_show=None,
    ypos=-0.6,
    xticks=None,
    ax=None,
    colors=None,
    at_risk_count_from_start_of_period=False,
    **kwargs
):
    """
    Add a table of counts below a survival or cumulative hazard plot showing how many individuals were
    at risk, censored, and experienced events at each specified time point, optionally using colored text.

    This function is an enhanced version of `lifelines.plotting.add_at_risk_counts`, with added support for:
    - Custom row selection (e.g., only "At risk" or all of ["At risk", "Censored", "Events"])
    - Colored text for each group's counts
    - Optional alignment with start or end of the time interval
    - Cleaner label formatting and multi-row text layout
    - Full compatibility with multiple fitters on the same axis

    Tip:
        It's often recommended to call ``plt.tight_layout()`` after this to avoid label clipping.

    Parameters
    ----------
    fitters : lifelines fitters
        One or more lifelines fitters (e.g., KaplanMeierFitter, NelsonAalenFitter) that have already been fitted.

    labels : iterable of str or bool, optional
        Custom labels for each fitter. If None, uses fitter._label. If False, no labels will be shown.

    rows_to_show : list of str, optional
        Subset of rows to show in the count table. Options are 'At risk', 'Censored', and 'Events'.
        Defaults to all three.

    ypos : float, optional (default: -0.6)
        Vertical position of the count table relative to the plot. Increase to move table higher.

    xticks : list of floats, optional
        Time points (x-axis values) at which counts should be shown. Defaults to visible x-axis ticks.

    ax : matplotlib.axes.Axes, optional
        Axes on which the survival curves are plotted. If None, uses current axes.

    colors : list of str or color codes, optional
        Colors for the groups’ text in the count table. Must be the same length as number of fitters.
        If None, defaults to black for all.

    at_risk_count_from_start_of_period : bool, optional (default: False)
        Whether the at-risk count should be taken from the start of the interval rather than the end
        (which is standard and used by KMunicate and lifelines).

    **kwargs :
        Additional keyword arguments passed to matplotlib's `text` rendering functions (e.g., fontsize).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object where the counts table was drawn.

    Examples
    --------
    >>> from lifelines import KaplanMeierFitter
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> kmf1 = KaplanMeierFitter().fit(data1, label="Group A")
    >>> kmf2 = KaplanMeierFitter().fit(data2, label="Group B")
    >>> kmf1.plot(ax=ax)
    >>> kmf2.plot(ax=ax)
    >>> add_at_risk_counts(kmf1, kmf2, colors=["blue", "orange"], ax=ax)
    >>> plt.tight_layout()

    References
    ----------
    Morris TP, Jarvis CI, Cragg W, et al. Proposals on Kaplan–Meier plots in medical research
    and a survey of stakeholder views: KMunicate. BMJ Open 2019;9:e030215.
    doi:10.1136/bmjopen-2019-030215
    """

    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()
    fig = kwargs.pop("fig", None)
    if fig is None:
        fig = plt.gcf()
    if labels is None:
        labels = [f._label for f in fitters]
    elif labels is False:
        labels = [None] * len(fitters)
    if rows_to_show is None:
        rows_to_show = ["At risk", "Censored", "Events"]
    else:
        assert all(
            row in ["At risk", "Censored", "Events"] for row in rows_to_show
        ), 'must be one of ["At risk", "Censored", "Events"]'
    n_rows = len(rows_to_show)

    # Create another axes where we can put size ticks
    ax2 = plt.twiny(ax=ax)
    # Move the ticks below existing axes
    # Appropriate length scaled for 6 inches. Adjust for figure size.
    ax_height = (
        ax.get_position().y1 - ax.get_position().y0
    ) * fig.get_figheight()  # axis height
    ax2_ypos = ypos / ax_height

    move_spines(ax2, ["bottom"], [ax2_ypos])
    # Hide all fluff
    remove_spines(ax2, ["top", "right", "bottom", "left"])
    # Set ticks and labels on bottom
    ax2.xaxis.tick_bottom()
    # Set limit
    min_time, max_time = ax.get_xlim()
    ax2.set_xlim(min_time, max_time)
    # Set ticks to kwarg or visible ticks
    if xticks is None:
        xticks = [xtick for xtick in ax.get_xticks() if min_time <= xtick <= max_time]
    ax2.set_xticks(xticks)
    # Remove ticks, need to do this AFTER moving the ticks
    remove_ticks(ax2, x=True, y=True)

    ticklabels = []

    for tick in ax2.get_xticks():
        lbl = ""

        # Get counts at tick
        counts = []
        for f in fitters:
            # this is a messy:
            # a) to align with R (and intuition), we do a subtraction off the at_risk column
            # b) we group by the tick intervals
            # c) we want to start at 0, so we give it it's own interval
            if at_risk_count_from_start_of_period:
                event_table_slice = f.event_table.assign(at_risk=lambda x: x.at_risk)
            else:
                event_table_slice = f.event_table.assign(
                    at_risk=lambda x: x.at_risk - x.removed
                )
            if not event_table_slice.loc[:tick].empty:
                event_table_slice = (
                    event_table_slice.loc[:tick, ["at_risk", "censored", "observed"]]
                    .agg(
                        {
                            "at_risk": lambda x: x.tail(1).values,
                            "censored": "sum",
                            "observed": "sum",
                        }
                    )  # see #1385
                    .rename(
                        {
                            "at_risk": "At risk",
                            "censored": "Censored",
                            "observed": "Events",
                        }
                    )
                    .fillna(0)
                )
                counts.extend([int(c) for c in event_table_slice.loc[rows_to_show]])
            else:
                counts.extend([0 for _ in range(n_rows)])
        if n_rows > 1:
            if tick == ax2.get_xticks()[0]:
                max_length = len(str(max(counts)))
                for i, c in enumerate(counts):
                    if i % n_rows == 0:
                        if is_latex_enabled():
                            lbl += (
                                ("\n" if i > 0 else "")
                                + r"\textbf{%s}" % labels[int(i / n_rows)]
                                + "\n"
                            )
                        else:
                            lbl += (
                                ("\n" if i > 0 else "")
                                + r"%s" % labels[int(i / n_rows)]
                                + "\n"
                            )
                    l = rows_to_show[i % n_rows]
                    s = (
                        "{}".format(l.rjust(10, " "))
                        + (" " * (max_length - len(str(c)) + 3))
                        + "{{:>{}d}}\n".format(max_length)
                    )

                    lbl += s.format(c)
            else:
                # Create tick label
                lbl += ""
                for i, c in enumerate(counts):
                    if i % n_rows == 0 and i > 0:
                        lbl += "\n\n"
                    s = "\n{}"
                    lbl += s.format(c)
        else:
            # if only one row to show, show in "condensed" version
            if tick == ax2.get_xticks()[0]:
                max_length = len(str(max(counts)))

                lbl += rows_to_show[0] + "\n"

                for i, c in enumerate(counts):
                    s = (
                        "{}".format(labels[i].rjust(10, " "))
                        + (" " * (max_length - len(str(c)) + 3))
                        + "{{:>{}d}}\n".format(max_length)
                    )
                    lbl += s.format(c)
            else:
                # Create tick label
                lbl += ""
                for i, c in enumerate(counts):
                    s = "\n{}"
                    lbl += s.format(c)
        ticklabels.append(lbl)

    if colors is None:
        colors = ['black'] * len(fitters)

    ax2.set_xticklabels([])
    # Text positioning
    line_height = 0.11  # adjust spacing between rows
    for xi, label in zip(ax2.get_xticks(), ticklabels):
        lines = label.split('\n')
        for li, line in enumerate(lines):
            if line=="At risk":
                line = "Number at risk"
                ax2.text(
                xi,
                ax2_ypos - li * line_height,
                line,
                color="black",
                ha="right",
                va="top",
                fontsize=kwargs.get("fontsize", plt.rcParams["font.size"])
                )
                continue
            if len(line)==0:
                continue
            color_idx = (li-1) % len(colors)  # match group order (li+1 because the first line is empty)
            ax2.text(
                xi,
                ax2_ypos - li * line_height,
                line,
                color=colors[color_idx],
                ha="right" if xi==0 else "center",
                va="top",
                fontsize=kwargs.get("fontsize", plt.rcParams["font.size"])
            )

    return ax