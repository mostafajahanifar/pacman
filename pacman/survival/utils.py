import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.plotting import (is_latex_enabled, move_spines, remove_spines,
                                remove_ticks)
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def find_optimal_threshold(
    risk_factor,
    time_to_event,
    event_occurred,
    n_thresholds: int = 100,
    min_pct: float = 0.05,
    min_group_size: int = 10,
    min_events: int = 5,
    verbose: bool = False
):
    """
    Find an optimal threshold on a continuous risk factor by maximizing separation
    in survival between low- vs high-risk groups via the log-rank test.

    Args:
        risk_factor: array-like (pd.Series or np.ndarray) of risk scores.
        time_to_event: array-like of same length, times to event or censoring.
        event_occurred: array-like of same length, 1 if event occurred, 0 if censored.
        n_thresholds: number of candidate thresholds to evaluate.
        min_pct: float in (0,0.5). Only consider thresholds between the min_pct and (1-min_pct)
                 quantiles of risk_factor to avoid extreme splits.
        min_group_size: minimum number of samples required in each group after split.
        min_events: minimum number of events required in each group for log-rank to be meaningful.
        verbose: if True, prints the chosen threshold and stats.

    Returns:
        optimal_threshold: float, the threshold on risk_factor that gave maximal log-rank statistic.
                           If no valid threshold found, returns None.
    """
    # Convert to pandas Series for convenience
    rf = pd.Series(risk_factor).dropna().reset_index(drop=True)
    tte = pd.Series(time_to_event).reset_index(drop=True)
    evt = pd.Series(event_occurred).reset_index(drop=True)

    # Align lengths and drop any indices where rf was NaN
    # (we assume tte and evt have same length as rf; if rf had NaN, we dropped those entries)
    # If there were NaNs in tte or evt, we keep them for now but logrank_test will error; could drop further if needed.
    if not (len(rf) == len(tte) == len(evt)):
        # If rf had NaNs removed, drop corresponding indices in tte, evt
        # But since we reset index on rf after dropna, better drop any where original rf was NaN:
        raise ValueError("After dropping NA in risk_factor, lengths do not match time/event arrays.")

    # If too few samples overall, bail out
    if len(rf) < 2 * min_group_size:
        if verbose:
            print(f"[find_optimal_threshold] Too few samples ({len(rf)}) for meaningful split.")
        return None

    # Determine candidate thresholds in the central range
    lower_q = rf.quantile(min_pct)
    upper_q = rf.quantile(1 - min_pct)
    if lower_q >= upper_q:
        # No range to search
        if verbose:
            print(f"[find_optimal_threshold] Quantile range invalid: lower_q={lower_q}, upper_q={upper_q}.")
        return None

    thresholds = np.linspace(lower_q, upper_q, n_thresholds)

    best_stat = -np.inf
    best_p = None
    best_thr = None

    for thr in thresholds:
        # Split into two groups: low (< thr) vs high (>= thr)
        mask_high = rf >= thr
        mask_low = ~mask_high

        # Check group sizes
        n_low = mask_low.sum()
        n_high = mask_high.sum()
        if n_low < min_group_size or n_high < min_group_size:
            continue

        # Check number of events in each
        events_low = evt[mask_low].sum()
        events_high = evt[mask_high].sum()
        if events_low < min_events or events_high < min_events:
            continue

        # Perform log-rank test
        try:
            res = logrank_test(
                tte[mask_low], tte[mask_high],
                event_observed_A=evt[mask_low],
                event_observed_B=evt[mask_high]
            )
        except Exception:
            continue

        stat = getattr(res, 'test_statistic', None)
        pval = getattr(res, 'p_value', None)
        if stat is None or pval is None:
            continue

        # Prefer larger test_statistic; if equal (within small tol), pick smaller p-value
        # Note: test_statistic may be negative if survival in high-risk unexpectedly better;
        # we take absolute value if we only care about separation regardless of direction.
        # Here we use absolute value:
        stat_abs = abs(stat)

        if stat_abs > best_stat or (np.isclose(stat_abs, best_stat) and pval < (best_p if best_p is not None else np.inf)):
            best_stat = stat_abs
            best_p = pval
            best_thr = thr

    if best_thr is None:
        if verbose:
            print("[find_optimal_threshold] No valid threshold found under constraints.")
        return None

    if verbose:
        print(f"[find_optimal_threshold] optimal threshold: {best_thr:.4f} → test_statistic={best_stat:.3f}, p-value={best_p:.3g}")

    return best_thr


def normalize_train_test(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list):
    """
    Normalize train and val DataFrames based on train mean and std for given feature_cols.
    Returns (train_norm, val_norm).
    """
    means = train[feature_cols].mean()
    stds = train[feature_cols].std().replace(0, 1.0)  # avoid division by zero
    train_norm = train.copy()
    val_norm = val.copy()
    train_norm[feature_cols] = (train[feature_cols] - means) / stds
    val_norm[feature_cols] = (val[feature_cols] - means) / stds
    return train_norm, val_norm

def approximate_hr_by_event_rate(T_lower, E_lower, T_upper, E_upper):
    """
    Approximate hazard ratio as ratio of event rates:
      (sum(E_upper) / sum(T_upper)) / (sum(E_lower) / sum(T_lower))
    Returns np.nan if denominator zero; returns np.inf if numerator >0 but denominator zero.
    """
    sum_events_lower = E_lower.sum()
    sum_time_lower = T_lower.sum()
    sum_events_upper = E_upper.sum()
    sum_time_upper = T_upper.sum()
    # avoid zero division
    rate_lower = sum_events_lower / sum_time_lower if sum_time_lower > 0 else np.nan
    rate_upper = sum_events_upper / sum_time_upper if sum_time_upper > 0 else np.nan

    if np.isnan(rate_lower) and np.isnan(rate_upper):
        return np.nan
    if np.isnan(rate_lower) and not np.isnan(rate_upper):
        return np.inf
    if not np.isnan(rate_lower) and np.isnan(rate_upper):
        return 0.0
    # both finite
    if rate_lower == 0:
        # if upper also zero, return nan; if upper>0, return inf
        if rate_upper == 0:
            return np.nan
        else:
            return np.inf
    return rate_upper / rate_lower

def survival_stratification_analysis(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feats_list,
    time_col: str,
    event_col: str,
    censor_at: float = None,
    cutoff_mode='mean',   # 'mean', 'median', 'quantile', 'optimal', or numeric
    cutoff_point: float = -1
):
    """
    Train (or handle single feature) on train_df, apply to val_df:
    - Normalize features by train mean/std.
    - If multiple features: fit CoxPH on train.
      If single feature: skip CoxPH fit; use normalized feature values directly.
    - Stratify validation set into high/low risk by cutoff derived from train partial risk (or raw feature).
    - Compute C-index on val set.
    - Compute p-value from log-rank test.
    - Compute hazard ratios:
        * multiple features: from CoxPH fitted on train
        * single feature: approximate via event-rate ratio between high vs low groups in train or val? 
          Here we compute from validation groups (so reflects test data).
    Returns:
        T_lower_test: pd.Series of times for low-risk group in val_df
        T_upper_test: pd.Series of times for high-risk group in val_df
        E_lower_test: pd.Series of events for low-risk group
        E_upper_test: pd.Series of events for high-risk group
        cindex_test: float
        pvalue_test: float
        test_hazard_ratios: pd.Series with index=feature name(s), values=HR or approximate HR
    """
    # 1. Prepare feature columns
    if isinstance(feats_list, str):
        feature_cols = [feats_list]
        use_cox = False
    else:
        feature_cols = list(feats_list)
        use_cox = True
    # Ensure time_col and event_col not in feature_cols
    feature_cols = [f for f in feature_cols if f not in (time_col, event_col)]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns provided after removing time/event columns.")

    required_cols = feature_cols + [time_col, event_col]

    # 2. Copy data to avoid side-effects
    train = train_df.copy()
    val = val_df.copy()

    # 3. Basic cleaning
    # Fill NaN with 0 (caution: may bias; user may choose other imputation)
    train = train.fillna(0)
    val = val.fillna(0)
    # Convert event_col >1 to 0
    train.loc[train[event_col] > 1, event_col] = 0
    val.loc[val[event_col] > 1, event_col] = 0
    # Censor validation times if requested
    if censor_at is not None and censor_at > 0:
        mask = val[time_col] > censor_at
        if mask.any():
            val.loc[mask, event_col] = 0
            val.loc[mask, time_col] = censor_at

    # 4. Check required columns
    missing_train = [col for col in required_cols if col not in train.columns]
    missing_val = [col for col in required_cols if col not in val.columns]
    if missing_train:
        raise ValueError(f"Training DataFrame missing columns: {missing_train}")
    if missing_val:
        raise ValueError(f"Validation DataFrame missing columns: {missing_val}")

    train_sub = train[required_cols].copy()
    val_sub = val[required_cols].copy()

    # 5. Normalize based on train
    train_norm, val_norm = normalize_train_test(train_sub, val_sub, feature_cols)

    # 6. Derive partial risk scores
    if not use_cox:
        # Single feature: use normalized feature values as risk score
        feat = feature_cols[0]
        partial_risk_train = train_norm[feat]
        partial_risk_test = val_norm[feat]
    else:
        # Multiple features: fit CoxPH on train_norm
        try:
            cph = CoxPHFitter(baseline_estimation_method='breslow', penalizer=0.001, l1_ratio=0.5)
            cph.fit(train_norm, duration_col=time_col, event_col=event_col)
        except Exception as e:
            raise RuntimeError(f"Failed to fit CoxPH on training data: {e}")
        partial_risk_train = cph.predict_partial_hazard(train_norm[feature_cols])
        partial_risk_test = cph.predict_partial_hazard(val_norm[feature_cols])

    # 7. Determine cutoff from train partial risk
    if cutoff_point is not None and cutoff_point >= 0:
        cutoff_value = cutoff_point
    else:
        # derive from cutoff_mode
        if isinstance(cutoff_mode, (int, float)):
            cutoff_value = float(cutoff_mode)
        else:
            mode = str(cutoff_mode).lower()
            if mode == 'mean':
                cutoff_value = partial_risk_train.mean()
            elif mode == 'median':
                cutoff_value = partial_risk_train.median()
            elif mode == 'quantile':
                cutoff_value = partial_risk_train.quantile(0.60)
            elif mode == 'optimal':
                # assume find_optimal_threshold exists
                try:
                    cutoff_value = find_optimal_threshold(
                        partial_risk_train,
                        train_norm[time_col],
                        train_norm[event_col]
                    )
                except Exception as e:
                    raise RuntimeError(f"Error in find_optimal_threshold: {e}")
            else:
                # try parse numeric
                try:
                    cutoff_value = float(cutoff_mode)
                except:
                    raise ValueError(f"Unrecognized cutoff_mode: {cutoff_mode}")

    # 8. On validation: stratify into high/low risk
    high_mask = partial_risk_test >= cutoff_value
    low_mask = ~high_mask

    T_upper_test = val_norm.loc[high_mask, time_col]
    E_upper_test = val_norm.loc[high_mask, event_col]
    T_lower_test = val_norm.loc[low_mask, time_col]
    E_lower_test = val_norm.loc[low_mask, event_col]

    # 9. Compute concordance index on validation
    try:
        # higher risk score => worse survival, so use -partial_risk_test
        cindex_test = concordance_index(val_norm[time_col], -partial_risk_test, val_norm[event_col])
    except Exception as e:
        raise RuntimeError(f"Failed to compute C-index on validation: {e}")

    # 10. Log-rank test between low vs high risk
    try:
        lr_res = logrank_test(T_lower_test, T_upper_test, E_lower_test, E_upper_test)
        pvalue_test = lr_res.p_value
    except Exception as e:
        raise RuntimeError(f"Failed log-rank test: {e}")

    # 11. Compute hazard ratios
    if len(feature_cols) == 1:
        # Approximate HR from event rates on validation groups
        hr_value = approximate_hr_by_event_rate(T_lower_test, E_lower_test, T_upper_test, E_upper_test)
        test_hazard_ratios = pd.Series({feature_cols[0]: hr_value})
    else:
        # Extract from fitted CoxPH on train
        # lifelines stores hazard_ratios_ as pd.Series
        test_hazard_ratios = cph.hazard_ratios_

    return T_lower_test, T_upper_test, E_lower_test, E_upper_test, cindex_test, pvalue_test, test_hazard_ratios

def add_at_risk_counts(
    *fitters,
    labels= None,
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