import argparse
import ast
import glob
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

from pacman.config import DATA_DIR, RESULTS_DIR, SURV_CANCERS
from pacman.survival.utils import (add_at_risk_counts,
                                   survival_stratification_analysis)

warnings.filterwarnings("ignore")

BOOTSTRAP_RUNS = 500
PERM_REPS = 1
CV_FOLDS = 5

# Selecting the feature(s) for patient stratification
feats_list = (
    "mean(ND)"  # if a list, will use CoxPH model, if not, raw feature values are used.
)

font_size = 12
fig_size = 5


def smart_text_location(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Get lines (KM curves)
    lines = ax.get_lines()

    # Get midpoints for comparison
    x_mid = (xlim[1] - xlim[0]) * 0.5
    y_threshold = (ylim[1] - ylim[0]) * 0.5

    # Count points in each quadrant
    lower_left_count = 0
    upper_right_count = 0

    for line in lines:
        xdata, ydata = line.get_data()
        lower_left_count += np.sum((xdata < x_mid) & (ydata < y_threshold))
        upper_right_count += np.sum((xdata > x_mid) & (ydata > y_threshold))

    return "lower left" if lower_left_count < upper_right_count else "upper right"


def cv_helper(
    discov_df,
    feats_list,
    event_col,
    time_col,
    split_folder,
    km_plot=False,
    x_label="Months",
    y_label=None,
    add_counts=False,
    shuffle_data=False,
    cutoff_mode="median",
):
    # running the cross-validation here
    EE = discov_df[event_col].to_numpy()
    rng = np.random.RandomState()
    c_indices = []
    p_values = []

    if split_folder is not None:
        ex_iterator = range(CV_FOLDS)
    else:
        ex_iterator = range(BOOTSTRAP_RUNS)

    for run in ex_iterator:
        # forming the training and validation cohorts
        if split_folder is None:  # bootstraping
            index_train = list(
                rng.choice(
                    np.nonzero(EE == 0)[0], size=len(EE) - np.sum(EE), replace=True
                )
            ) + list(rng.choice(np.nonzero(EE == 1)[0], size=np.sum(EE), replace=True))
            index_test = list(set(range(len(EE))).difference(index_train))
            train_set = discov_df.iloc[index_train]
            test_set = discov_df.iloc[index_test]
        else:
            split_path = f"{split_folder}/splits_{run}.csv"
            split_df = pd.read_csv(split_path)
            train_set = discov_df[
                discov_df["bcr_patient_barcode"].isin(split_df["train"])
            ]
            test_set = discov_df[discov_df["bcr_patient_barcode"].isin(split_df["val"])]
        train_set.reset_index(inplace=True)
        test_set.reset_index(inplace=True)

        if shuffle_data:  # for permutation test
            random_indices = np.random.permutation(train_set.index)
            train_set[event_col] = (
                train_set[event_col].loc[random_indices].reset_index(drop=True)
            )
            train_set[time_col] = (
                train_set[time_col].loc[random_indices].reset_index(drop=True)
            )

            random_indices = np.random.permutation(test_set.index)
            test_set[event_col] = (
                test_set[event_col].loc[random_indices].reset_index(drop=True)
            )
            test_set[time_col] = (
                test_set[time_col].loc[random_indices].reset_index(drop=True)
            )

        # Normalizing the datasets and running the model
        output = survival_stratification_analysis(
            train_set,
            test_set,
            feats_list,
            time_col,
            event_col,
            censor_at,
            cutoff_mode,
            cutoff_point,
        )

        # accumulate the runs outputs
        if run == 0:
            (
                T_lower_test,
                T_upper_test,
                E_lower_test,
                E_upper_test,
                cindex_test,
                pvalue_test,
                test_hazard_ratios,
            ) = output
        if run != 0 and output != -1:
            (
                _T_lower_test,
                _T_upper_test,
                _E_lower_test,
                _E_upper_test,
                cindex_test,
                pvalue_test,
                temp,
            ) = output
            T_lower_test = pd.concat(
                [T_lower_test, _T_lower_test],
                axis=0,
                join="outer",
                ignore_index=True,
                sort=False,
            )
            T_upper_test = pd.concat(
                [T_upper_test, _T_upper_test],
                axis=0,
                join="outer",
                ignore_index=True,
                sort=False,
            )
            E_lower_test = pd.concat(
                [E_lower_test, _E_lower_test],
                axis=0,
                join="outer",
                ignore_index=True,
                sort=False,
            )
            E_upper_test = pd.concat(
                [E_upper_test, _E_upper_test],
                axis=0,
                join="outer",
                ignore_index=True,
                sort=False,
            )
            test_hazard_ratios = pd.concat(
                [test_hazard_ratios, temp],
                axis=1,
                join="outer",
                ignore_index=True,
                sort=False,
            )
        c_indices.append(cindex_test)
        p_values.append(pvalue_test)

    df_to_save = test_hazard_ratios.T
    df_to_save["c_index"] = c_indices
    df_to_save["p_value"] = p_values

    logrank_results = logrank_test(
        T_lower_test, T_upper_test, E_lower_test, E_upper_test
    )

    if km_plot:
        fig = plt.figure(
            figsize=(fig_size - 1.8, fig_size - 2)
        )  ##adjust according to font size
        ax = fig.add_subplot(111)
        ax.set_xlabel("", fontsize=font_size)
        ax.set_ylabel("", fontsize=font_size)
        ax.tick_params(axis="x", labelsize=font_size)
        ax.tick_params(axis="y", labelsize=font_size)

        # Initializing the KaplanMeierModel for each group
        km_upper = KaplanMeierFitter()
        km_lower = KaplanMeierFitter()

        # labels
        if isinstance(feats_list, str):
            upper_label = f"{feats_list}-High"
            lower_label = f"{feats_list}-Low"
        else:
            upper_label = "High-Risk"
            lower_label = "Low-Risk"

        ax = km_upper.fit(
            T_upper_test, event_observed=E_upper_test, label=upper_label
        ).plot_survival_function(
            ax=ax,
            show_censors=True,
            censor_styles={"ms": 4},
            color="r",
            ci_show=False,
            xlabel=x_label,
            ylabel=y_label,
        )
        ax = km_lower.fit(
            T_lower_test, event_observed=E_lower_test, label=lower_label
        ).plot_survival_function(
            ax=ax,
            show_censors=True,
            censor_styles={"ms": 4},
            color="b",
            ci_show=False,
            xlabel=x_label,
            ylabel=y_label,
        )
        ax.get_legend().remove()

        if add_counts:
            fig_copy = plt.figure(figsize=(fig_size - 1, fig_size - 2))
            ax_copy = fig_copy.add_subplot(111)
            ax_copy.set_xlabel("", fontsize=font_size)
            ax_copy.set_ylabel("", fontsize=font_size)
            ax_copy.tick_params(axis="x", labelsize=font_size)
            ax_copy.tick_params(axis="y", labelsize=font_size)

            # Initializing the KaplanMeierModel for each group
            ax_copy = km_upper.fit(
                T_upper_test, event_observed=E_upper_test, label=upper_label
            ).plot_survival_function(
                ax=ax_copy,
                show_censors=True,
                censor_styles={"ms": 5},
                color="r",
                ci_show=False,
                xlabel=x_label,
                ylabel=y_label,
            )
            ax_copy = km_lower.fit(
                T_lower_test, event_observed=E_lower_test, label=lower_label
            ).plot_survival_function(
                ax=ax_copy,
                show_censors=True,
                censor_styles={"ms": 5},
                color="b",
                ci_show=False,
                xlabel=x_label,
                ylabel=y_label,
            )

            add_at_risk_counts(
                km_lower,
                km_upper,
                rows_to_show=["At risk"],
                xticks=[0, 25, 50, 75, 100],
                ypos=-0.6,
                colors=["blue", "red"],
                ax=ax_copy,
                fig=fig_copy,
                fontsize=int(font_size * 1),
            )
            fig_copy.subplots_adjust(bottom=0.4)
            fig_copy.subplots_adjust(left=0.2)
            ax_copy.get_legend().remove()

            return df_to_save, logrank_results, fig, fig_copy

        return df_to_save, logrank_results, fig

    return df_to_save, logrank_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--studies", default=["all"], nargs="+")
    parser.add_argument("--censor_at", default=120, type=int)
    parser.add_argument("--splits_root", default="./splits/")
    parser.add_argument("--cutoff_mode", default="median")
    parser.add_argument("--cutoff_point", default=-1, type=int)  # manual cuttof point

    args = parser.parse_args()

    print(7 * "=" * 7)
    print("Patient stratification with cross-valitated KM plots")
    print(f"Using features: {feats_list}")
    print(7 * "=" * 7)

    studies = args.studies
    censor_at = args.censor_at
    splits_root = args.splits_root
    cutoff_mode = args.cutoff_mode
    cutoff_point = args.cutoff_point  ## if set to -1 then median will be used

    results_root = f"{RESULTS_DIR}/survival/kmplots_cv/"
    mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))
    print(args)

    for event_type in ["DSS", "PFI", "OS", "DFI"]:
        time_col = f"{event_type}.time"
        event_col = event_type

        if studies == ["all"]:  # all studies are processed one-by-one
            items = SURV_CANCERS
        else:  # multiple studies are combined
            items = ["".join(studies).upper()]

        for study in items:
            print(f"*** Working on {study} study in {event_type} endpoint ***")
            try:
                discov_df = mitosis_feats.loc[mitosis_feats["type"] == study]
                discov_df = discov_df.dropna(subset=[event_col, time_col])
                discov_df[event_col] = discov_df[event_col].astype(int)
                discov_df[time_col] = (discov_df[time_col] / 30.4).astype(int)
                print(
                    f"Number of cases in FS experiment after dropping NA: {len(discov_df)}"
                )

                # setting the results path
                save_dir = f"{results_root}/{study}/"
                os.makedirs(save_dir, exist_ok=True)

                if splits_root is not None:
                    split_folder = os.path.join(splits_root, event_type, study)
                else:
                    split_folder = None

                # running the cross-validation for the real data
                df_to_save, logrank_results, km_fig, km_fig_counts = cv_helper(
                    discov_df,
                    feats_list,
                    event_col,
                    time_col,
                    split_folder,
                    km_plot=True,
                    add_counts=True,
                    cutoff_mode=cutoff_mode,
                )
                ref_logrank_stat = logrank_results.test_statistic

                # now repeat the cross-validation several times and check the statistics to arrive the p-value
                num_greater_by_chance = 0
                num_valid_runs = 0
                for _ in tqdm(range(PERM_REPS), desc="p-value permutation tests"):
                    try:
                        _, logrank_results_repeats = cv_helper(
                            discov_df,
                            feats_list,
                            event_col,
                            time_col,
                            split_folder,
                            km_plot=False,
                            shuffle_data=True,
                            cutoff_mode=cutoff_mode,
                        )
                    except:
                        continue
                    if logrank_results_repeats.test_statistic > ref_logrank_stat:
                        num_greater_by_chance += 1
                    num_valid_runs += 1
                corrected_p_value = num_greater_by_chance / num_valid_runs

                avr_cindex = df_to_save["c_index"].mean()
                print("Average C-Index: ", avr_cindex)
                print("Std C-Index: ", df_to_save["c_index"].std())
                print("Corrected p-value: ", corrected_p_value)

                # adding the corrected p-value to the figure
                if corrected_p_value < 0.0001:
                    pvalue_txt = "p < 0.0001"
                else:
                    pvalue_txt = "p = " + str(np.round(corrected_p_value, 4))
                ax = km_fig.axes[0]  # Get the ax object from fig
                loc = smart_text_location(ax)
                ax.add_artist(
                    AnchoredText(
                        pvalue_txt, loc=loc, frameon=False, prop=dict(size=font_size)
                    )
                )
                # ax.add_artist(AnchoredText(pvalue_txt, loc='lower left', frameon=False, prop=dict(size=font_size)))
                ax.set_ylabel(f"{event_type.upper()} Probability")
                ax.set_ylim(0, 1)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xlim(0, censor_at + 1)
                ax.set_title("".join(study).upper(), fontsize=font_size + 2)
                ax.spines[["right", "top"]].set_visible(False)

                # save the results
                save_path = (
                    save_dir
                    + f"kmplot_cv_{study}_{event_type}_censor{censor_at}_cindex{avr_cindex:.2}_pvalue{corrected_p_value:.3}"
                )
                df_to_save.to_csv(save_path + ".csv", index=None)
                km_fig.savefig(
                    save_path + ".png", dpi=600, bbox_inches="tight", pad_inches=0.01
                )
                km_fig.savefig(
                    save_path + ".pdf", dpi=600, bbox_inches="tight", pad_inches=0.01
                )

                # save the km figure with counts
                ax = km_fig_counts.axes[0]  # Get the ax object from fig
                # ax.add_artist(AnchoredText(pvalue_txt, loc='lower left', frameon=False, prop=dict(size=font_size)))
                ax.add_artist(
                    AnchoredText(
                        pvalue_txt, loc=loc, frameon=False, prop=dict(size=font_size)
                    )
                )
                ax.set_ylabel(f"{event_type.upper()} Probability")
                ax.set_ylim(0, 1)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xlim(0, censor_at + 1)
                ax.set_title("".join(study).upper(), fontsize=font_size + 2)
                ax.spines[["right", "top"]].set_visible(False)

                # save the results
                save_path = (
                    save_dir
                    + f"risked_kmplot_cv_{study}_{event_type}_censor{censor_at}_cindex{avr_cindex:.2}_pvalue{corrected_p_value:.3}"
                )
                km_fig_counts.savefig(
                    save_path + ".png", dpi=600, bbox_inches="tight", pad_inches=0.01
                )
            except Exception as e:
                print(
                    f"! Failed working on {study} study in {event_type} endpoint: {e}"
                )
