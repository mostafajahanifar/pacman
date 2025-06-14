import argparse
import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR

# ignore all types of warnings
warnings.filterwarnings("ignore", category=UserWarning)

# get an argument parser to set the mode to pan-cancer or cancer-specific
parser = argparse.ArgumentParser(description="Immune Subtypes Distribution Comparison Analysis")
parser.add_argument("--mode", type=str, default="Pan-cancer", choices=["pan-cancer", "all"],
                    help="Mode of analysis: 'Pan-cancer' for all cancers together or 'all' for specific cancer types separately")
args = parser.parse_args()


print (7 * "=" * 7)
print(f"Running Immune Subtypes Distribution Comparison Analysis in {args.mode} mode")
print(7 * "=" * 7)


save_root = f"{RESULTS_DIR}/immune/distributions/"
os.makedirs(save_root, exist_ok=True)

selected_feats = [
    "HSC",
    "mean(ND)",
    "cv(ND)",
    "mean(CL)",
    "mean(HC)",
    # the following features for morphology analyses
    "AMAH",
    "AFW",
]

#reading necessary data
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]+selected_feats]

immune_df = pd.read_csv(os.path.join(DATA_DIR, "tcga_all_immune_new.csv"))
# drop missing data
immune_df = immune_df.dropna(axis=1, how="all")

if args.mode.lower() == "all":
    items = ["Pan-cancer"]+ALL_CANCERS
else:
    items = ["Pan-cancer"]

for cancer_type in items:
    print(cancer_type)

    if cancer_type=="Pan-cancer":
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"].isin(ALL_CANCERS)]
    else:
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"]==cancer_type]

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

    if "Immune Subtype" not in df2_common.columns:
        print(f"Not many cases with immune subtype in {cancer_type}: {len(df2_common)}")
        continue

    subtype_sorted = df2_common.sort_values(by="Immune Subtype")["Immune Subtype"]
    subtype_sorted = subtype_sorted.dropna(axis=0, how="any")
    working_df = pd.concat([df1_common[selected_feats], df2_common["Immune Subtype"]], axis=1)
    working_df = working_df.iloc[subtype_sorted.index]

    # Define the custom colors for each immune subtype
    colors = {
        'C1': '#FF0000',
        'C2': '#FFFA00',
        'C3': '#00FD00',
        'C4': '#00FFFF',
        'C5': '#0039FF',
        'C6': '#FF25FF',
    }

    save_dir = f"{save_root}/{cancer_type}"
    os.makedirs(save_dir, exist_ok=True)

    # Create the boxplot
    for mit_feat in selected_feats:
        try:
            plt.figure(figsize=(2, 2))
            ax = sns.boxplot(data=working_df, x="Immune Subtype", y=mit_feat, palette=colors, showfliers=False, linewidth=0.5)

            # Perform pairwise statistical tests
            subtypes = working_df['Immune Subtype'].unique()
            pairs = list(itertools.combinations(subtypes, 2))
            significant_pairs = []

            # show only non-significant ones
            for (subtype1, subtype2) in pairs:
                data1 = working_df[working_df['Immune Subtype'] == subtype1][mit_feat]
                data2 = working_df[working_df['Immune Subtype'] == subtype2][mit_feat]
                stat, p_value = mannwhitneyu(data1, data2)
                if p_value > 0.01:
                    significant_pairs.append((subtype1, subtype2))

            annotator = Annotator(ax, significant_pairs, data=working_df, x="Immune Subtype", y=mit_feat)
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=False, verbose=2)
            annotator.apply_and_annotate()

            
            plt.savefig(f"{save_dir}/mitosis_{mit_feat}_in_immunue_subtypes.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
        except Exception as e:
            print(f"Error BOX plotting {mit_feat} in immune subtypes for {cancer_type}: {e}")

    # plot hot/cold population in immune subtypes
    try:
        working_df = pd.concat([df1_common["temperature"], df2_common["Immune Subtype"]], axis=1)
        working_df = working_df.dropna(axis=0, how="any")
        working_df = working_df.rename(columns={"temperature": "Mitotic Temperature"})
        # Group by 'Immune Subtype' and 'cluster_2', then count occurrences
        grouped = working_df.groupby(['Immune Subtype', 'Mitotic Temperature']).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(2, 2))
        grouped.plot(kind='bar', ax=ax, cmap="coolwarm")

        # annotate the bars with Mitotic Hot/Cold Ratio
        # Function to simplify ratios
        def simplify_ratio(a, b):
            divisor = min(a,b)
            return f'{round(a/divisor)}:{round(b/divisor)}'
        grouped['Ratio'] = grouped.apply(lambda row: simplify_ratio(row.get('Cold', 0), row.get('Hot', 0)), axis=1)
        # Add the calculated approximate ratio on top of each bar group
        for idx, (immune_subtype, row) in enumerate(grouped.iterrows()):
            ratio = row['Ratio']
            max_height = max(row['Hot'], row['Cold']) if 'Hot' in row and 'Cold' in row else 0
            ax.text(idx, max_height + 0.5, ratio, ha='center', va='bottom')


        ax.set_xlabel('Immune Subtype')
        ax.set_ylabel('Population')
        ax.set_ylim([0, 1900])
        plt.legend(title="Mitosis", bbox_to_anchor=(1.01, 1), loc='upper left')
        # ax.set_title('Population of Each Immune Subtype for Each Cluster')
        plt.xticks(rotation=0)
        plt.savefig(f"{save_dir}/population_Hot-Cold_in_immunue_subtypes.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
    except Exception as e:
        print(f"Error plotting population of Hot-Cold in immune subtypes for {cancer_type}: {e}")


    # Violing plot of immune charachtersics for hot/cold categories
    immune_feats = [
                    # selected tumor properties (immune, genomic, and other features)
                    "CTA Score",
                    "Stromal Fraction", "Intratumor Heterogeneity", "Proliferation",
                    "SNV Neoantigens", "Indel Neoantigens", "Nonsilent Mutation Rate",
                    "Number of Segments", "Fraction Altered", "Aneuploidy Score",
                    "Homologous Recombination Defects",
                    "Th1:Th2 cells ratio", "TIL Regional Fraction", "Wound Healing",
                    "Lymphocyte Infiltration Signature Score", "IFN-gamma Response", "TGF-beta Response",
                    ]

    for immune_feat in immune_feats:
        try:
            working_df = pd.concat([df1_common["temperature"], df2_common[immune_feat]], axis=1)
            working_df = working_df.dropna(axis=0, how="any")
            working_df = working_df.rename(columns={"temperature": "Mitotic Temperature"})
            working_df = working_df.sort_values(by="Mitotic Temperature")


            #standardize the values
            if immune_feat in ["SNV Neoantigens", "Indel Neoantigens", "Nonsilent Mutation Rate", "Number of Segments"]:
                working_df[immune_feat] = working_df[immune_feat].apply(lambda x: np.log10(x + 1))
                working_df = working_df.rename(columns={immune_feat: f"-log10({immune_feat}+1)"})
                immune_feat = f"-log10({immune_feat}+1)"

            # Unique temperature states
            temperature_states = working_df['Mitotic Temperature'].unique()

            # Create the boxplot using matplotlib
            plt.figure(figsize=(1.8,3))
            ax = plt.gca()

            # Boxplot
            # Get the colors from the 'copper' palette
            palette = sns.color_palette('coolwarm', n_colors=100)
            # Set the alpha value
            alpha = 0.1
            # Modify the colors to include alpha
            colors = [(r, g, b, alpha) for r, g, b in palette]
            # Assign the first and last colors from the palette
            colors = [colors[0], colors[-1]]

            sns.violinplot(
                data=working_df,
                x='Mitotic Temperature',
                y=immune_feat,
                ax=ax,
                palette=colors,
                inner="box",
                linewidth=0.5,
            )

            sns.boxplot(
                data=working_df,
                x='Mitotic Temperature',
                y=immune_feat,
                ax=ax,
                showcaps=True,
                boxprops=dict(facecolor=(1, 1, 1, 0), edgecolor='black', linewidth=0.5),
                whiskerprops=dict(color='black', linewidth=0.5),
                capprops=dict(color='black', linewidth=0.5),
                medianprops=dict(color='black', linewidth=0.5),
                showfliers=False,
                linewidth=0.5,
                whis=0,
                zorder=10  # ensure it draws on top
            )

            # Perform pairwise statistical tests and add annotations
            pairs = list(itertools.combinations(temperature_states, 2))
            y_max = working_df[immune_feat].max()


            # # Perform pairwise statistical tests
            annotator = Annotator(ax, pairs, data=working_df, x='Mitotic Temperature', y=immune_feat, plot="violinplot")
            p_val_fmt = None # [[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', pvalue_format=p_val_fmt, hide_non_significant=False, verbose=2, alpha=0.01,
                                show_test_name=False)
            annotator.apply_and_annotate()

            ax.set_xlabel(None)
            if immune_feat == "Proliferation":
                ax.set_ylim([-3, 3])
            if immune_feat == "CTA Score":
                ax.set_ylim([-1, 10.5])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Customize the plot
            plt.xticks(rotation=0)
            if cancer_type != "Pan-cancer":
                plt.title(cancer_type)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/immune_{immune_feat}_in_Hot-Cold.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
        except Exception as e:
            print(f"Error VIOLIN plotting immune feature {immune_feat} in {cancer_type}: {e}")