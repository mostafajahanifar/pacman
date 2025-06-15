import os

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter

from pacman.config import DATA_DIR, RESULTS_DIR, SURV_CANCERS

save_root = f"{RESULTS_DIR}/morphology/survival_univariate_hr/"
os.makedirs(save_root, exist_ok=True)


mit_temp = "all"
valid_cancer_for_event = {
    "PFI": ['GBMLGG', 'SKCM', 'LUAD', 'HNSC', 'LIHC', 'BLCA', 'COADREAD', 'KIRC', 'BRCA', 'LUSC', 'STAD', 'SARC', 'UCEC', 'PAAD', 'ESCA', 'OV', 'CESC', 'KIRP', 'MESO', 'TGCT', 'UCS', 'ACC', 'PCPG'],
    "OS": ['GBMLGG', 'HNSC', 'LUSC', 'SKCM', 'BLCA', 'KIRC', 'LUAD', 'BRCA', 'STAD', 'LIHC', 'COADREAD', 'PAAD', 'SARC', 'UCEC', 'OV', 'ESCA', 'CESC', 'MESO', 'UCS', 'ACC', 'DLBC'],
    "DSS": ['GBMLGG', 'SKCM', 'HNSC', 'BLCA', 'KIRC', 'LUAD', 'STAD', 'LUSC', 'LIHC', 'BRCA', 'PAAD', 'COADREAD', 'SARC', 'OV', 'UCEC', 'CESC', 'ESCA', 'MESO', 'UCS', 'ACC'],
    "DFI": ['LIHC', 'LUAD', 'BRCA', 'LUSC', 'SARC', 'UCEC', 'STAD', 'BLCA', 'TGCT', 'OV', 'PAAD', 'ESCA'],
}

# Function to perform univariate analysis and extract HR, CI, and p-value
def cox_univariate_hr_analysis(df, feature, event_col, time_col):
    cph = CoxPHFitter()
    data = df[[feature, time_col, event_col]].dropna(how="any")  # Remove missing values

    # Normalize feature
    data[feature] = (data[feature] - data[feature].mean()) / (data[feature].std() + 1e-9)

    # Fit Cox model
    cph.fit(data, duration_col=time_col, event_col=event_col)

    # Extract HR, CI, and p-value
    hr = cph.summary.loc[feature, 'exp(coef)']
    ci_lower = cph.summary.loc[feature, 'exp(coef) lower 95%']
    ci_upper = cph.summary.loc[feature, 'exp(coef) upper 95%']
    p_value = cph.summary.loc[feature, 'p']

    return hr, ci_lower, ci_upper, p_value

# Function to perform univariate analysis across different event types and cancer types
def analyze_univariate_hr_by_event_and_cancer(df, selected_feats):
    results = {}
    event_types = ["OS", "PFI", "DSS", "DFI"]

    for event_type in event_types:
        event_col = event_type
        time_col = f"{event_type}.time"
        results[event_type] = {}
        cancer_types = sorted(valid_cancer_for_event[event_type])

        for cancer in cancer_types:
            print(f"{event_type} - {cancer}")
            cancer_df = df[df['type'] == cancer]

            if mit_temp in ["Hot", "Cold"]:
                cancer_df = cancer_df[cancer_df["temperature"] == mit_temp]

            feature_stats = {}
            for feat in selected_feats:
                try:
                    hr, ci_lower, ci_upper, p_value = cox_univariate_hr_analysis(cancer_df, feat, event_col, time_col)
                    feature_stats[feat] = (hr, ci_lower, ci_upper, p_value)
                except Exception as e:
                    print(f"Failed {event_type} - {cancer} - {feat}: because: {e}")
                    feature_stats[feat] = None

            results[event_type][cancer] = feature_stats

    return results

# Function to plot hazard ratios with confidence intervals using dot plots (significant using stars)
def plot_hr_with_ci_dotplots(results, selected_feats, mode="selected"):
    for event_type, cancers in results.items():
        cancer_types = list(cancers.keys())
        
        # If mode is "selected", filter out cancer types with no significant p-values
        if mode == "selected":
            cancer_types = [
                cancer for cancer in cancer_types
                if any(cancers[cancer][feat] and cancers[cancer][feat][3] <= 0.05 for feat in selected_feats)
            ]
        
        num_cancers = len(cancer_types)
        num_feats = len(selected_feats)

        # Create a subplot grid with cancer types as rows and features as columns
        fig, axs = plt.subplots(num_cancers, num_feats, figsize=(num_feats*2, num_cancers/3.5), sharex=True, sharey=True)
        
        for i, cancer in enumerate(cancer_types):
            for j, feat in enumerate(selected_feats):
                ax = axs[i, j] if num_cancers > 1 else axs[j]
                stats = cancers[cancer].get(feat)

                if stats:
                    hr, ci_lower, ci_upper, p_value = stats

                    # Plot dot at HR value with error bars
                    color = 'blue' if p_value <= 0.05 else 'darkgray'
                    ax.errorbar(hr, 0, xerr=[[hr - ci_lower], [ci_upper - hr]], fmt='o', color=color, capsize=4)

                    # If p-value is significant, add a star symbol before the error bar
                    if p_value <= 0.05:
                        ax.text(hr + (ci_upper - hr) + 0.1, 0, '*', fontsize=12, color='black', verticalalignment='center')

                # Customize the subplot appearance
                # if i == num_cancers - 1:
                #     ax.set_xlabel("HR")
                if i == 0:
                    ax.set_title(feat)
                if j == 0:
                    ax.set_ylabel(cancer, rotation=0, horizontalalignment='right', verticalalignment='center')

                # Set y-limits and remove all y-axis ticks except at y=0
                ax.set_yticks([0])  # Only show tick at y=0

                # Remove y-axis tick labels
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

                if i != num_cancers - 1:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                # Remove spines for cleaner look, except for leftmost and bottommost subplots
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                if i == num_cancers - 1:
                    ax.spines['bottom'].set_visible(True)

                if j == 0:
                    ax.spines['left'].set_visible(True)

                ax.axvline(x=1, color='lightgray', linestyle='--', linewidth=1, zorder=0)  # Line at HR=1
                ax.set_xlim([0, 3])
                ax.set_xticks([1, 2])

        # Adjust layout and save the plot with mode in filename
        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.savefig(f"{save_root}/hr_{mode}_{mit_temp}_{event_type}.png", dpi=600, bbox_inches='tight', pad_inches=0.01)

# Load your data
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))

selected_feats = [
    "AMAH",
    "AFW",
    "AMH",
    "HSC",
]

# Run the univariate analysis and plot the hazard ratios with confidence intervals
univariate_hr_results = analyze_univariate_hr_by_event_and_cancer(mitosis_feats, selected_feats)
plot_hr_with_ci_dotplots(univariate_hr_results, selected_feats)
