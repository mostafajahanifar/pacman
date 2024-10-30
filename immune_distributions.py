import os, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from utils import featre_to_tick, get_colors_dict
import argparse
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm
import seaborn as sns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import itertools
from statannotations.Annotator import Annotator
from math import gcd

save_root = "results_final_all/immune/distributions/"
# ALL_CANCERS = ['BRCA', 'KIRC', 'UCEC', 'LGG', 'LUSC', 'LUAD', 'HNSC', 'COADREAD', 'SKCM',
#                 'GBM', 'BLCA', 'STAD', 'LIHC', 'KIRP', 'CESC', 'PAAD', 'ESCA', 'PCPG', 'KICH', 'OV']
ALL_CANCERS = ['SARC',
    'LIHC',
    'THYM',
    'ACC',
    'BRCA',
    'KICH',
    'STAD',
    'BLCA',
    'THCA',
    'GBMLGG',
    'UCEC',
    'LUAD',
    'KIRC',
    'KIRP',
    'PAAD',
    'CESC',
    'PCPG',
    'MESO',
    'SKCM',
    'PRAD',
    'COADREAD',
    'ESCA',
    'LUSC',
    'HNSC',
    'OV',
    'TGCT',
    'CHOL',
    'DLBC',
    'UCS'
 ]
ALL_CANCERS = sorted(ALL_CANCERS)

# keep only columns that are related to mutations
immune_df = pd.read_excel("gene/data/tcga_all_immune.xlsx")
immune_df["TCGA Study"] = immune_df["TCGA Study"].replace(["COAD", "READ"], "COADREAD")
immune_df["TCGA Study"] = immune_df["TCGA Study"].replace(["GBM", "LGG"], "GBMLGG")

selected_feats = [
    "mit_hotspot_count",
    "mit_nodeDegrees_mean",
    "mit_nodeDegrees_cv",
    "mit_clusterCoff_mean",
    "mit_cenHarmonic_mean",
    "aty_ahotspot_count",
    "aty_wsi_ratio",
]

mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancer_withAtypical.csv')
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]+selected_feats]
mitosis_feats.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type", "temperature"] else col for col in mitosis_feats.columns]
mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")

for cancer_type in ['Pan-cancer']: #sorted(["Pan-cancer"]+ALL_CANCERS):
    print(cancer_type)

    if cancer_type=="Pan-cancer":
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"].isin(ALL_CANCERS)]
        gene_exp_cancer = immune_df[immune_df["TCGA Study"].isin(ALL_CANCERS)]
    else:
        mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"]==cancer_type]
        gene_exp_cancer = immune_df[immune_df["TCGA Study"]==cancer_type]


    # drop missing mutations
    gene_exp_cancer = gene_exp_cancer.dropna(axis=1, how="all")
    # drop cases with all mutations as Nan


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

    if "Immune Subtype" not in df2_common.columns:
        print(f"{cancer_type}: {len(df2_common)}")
        continue

    subtype_sorted = df2_common.sort_values(by="Immune Subtype")["Immune Subtype"]
    subtype_sorted = subtype_sorted.dropna(axis=0, how="any")
    mosi = pd.concat([df1_common[["HSC", "mean(ND)", "cv(ND)", "mean(CL)", "mean(HC)", "AMAH", "AFW"]], df2_common["Immune Subtype"]], axis=1)
    mosi = mosi.iloc[subtype_sorted.index]

    # Define the custom colors for each immune subtype
    colors = {
        'C1': '#FF0000',
        'C2': '#FFFA00',
        'C3': '#00FD00',
        'C4': '#00FFFF',
        'C5': '#0039FF',
        'C6': '#FF25FF',
    }

    # Create the boxplot
    for mit_feat in ["HSC", "mean(ND)", "cv(ND)", "mean(CL)", "mean(HC)", "AMAH", "AFW"]:
        try:
            plt.figure(figsize=(2, 2))
            ax = sns.boxplot(data=mosi, x="Immune Subtype", y=mit_feat, palette=colors, showfliers=False, linewidth=0.5)

            # Perform pairwise statistical tests
            subtypes = mosi['Immune Subtype'].unique()
            pairs = list(itertools.combinations(subtypes, 2))
            significant_pairs = []

            # show only non-significant ones
            for (subtype1, subtype2) in pairs:
                data1 = mosi[mosi['Immune Subtype'] == subtype1][mit_feat]
                data2 = mosi[mosi['Immune Subtype'] == subtype2][mit_feat]
                stat, p_value = mannwhitneyu(data1, data2)
                if p_value > 0.01:
                    significant_pairs.append((subtype1, subtype2))

            annotator = Annotator(ax, significant_pairs, data=mosi, x="Immune Subtype", y=mit_feat)
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', hide_non_significant=False, verbose=2)
            annotator.apply_and_annotate()

            save_dir = f"{save_root}/{cancer_type}"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/mitosis_{mit_feat}_in_immunue_subtypes.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
        except:
            print("not working for : ", mit_feat)

    # plot hot/cold population in immune subtypes
    try:
        mosi = pd.concat([df1_common["temperature"], df2_common["Immune Subtype"]], axis=1)
        mosi = mosi.dropna(axis=0, how="any")
        mosi = mosi.rename(columns={"temperature": "Mitotic Temperature"})
        # Group by 'Immune Subtype' and 'cluster_2', then count occurrences
        grouped = mosi.groupby(['Immune Subtype', 'Mitotic Temperature']).size().unstack(fill_value=0)

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
    except:
        print("Population plotting fails")


    # Violing plot of immune charachtersics for hot/cold categories
    immune_feats = ["Stromal Fraction", "Intratumor Heterogeneity", "Proliferation",
                    "SNV Neoantigens", "Indel Neoantigens", "Nonsilent Mutation Rate",
                    "Number of Segments", "Fraction Altered", "Aneuploidy Score",
                    "Homologous Recombination Defects", "CTA Score",
                    # immune features
                    "Leukocyte Fraction", "TIL Regional Fraction", "Wound Healing", "Macrophage Regulation",
                    "Lymphocyte Infiltration Signature Score", "IFN-gamma Response", "TGF-beta Response",
                    "BCR Evenness", "TCR Evenness", "Th1 Cells", "Th2 Cells", "Th17 Cells"]
    
    # immune_feats = ["Proliferation",]
    for immune_feat in immune_feats:
        try:
            mosi = pd.concat([df1_common["temperature"], df2_common[immune_feat]], axis=1)
            mosi = mosi.dropna(axis=0, how="any")
            mosi = mosi.rename(columns={"temperature": "Mitotic Temperature"})
            mosi = mosi.sort_values(by="Mitotic Temperature")

            #standardize the values
            if immune_feat in ["SNV Neoantigens", "Indel Neoantigens", "Nonsilent Mutation Rate", "Number of Segments"]:
                mosi[immune_feat] = mosi[immune_feat].apply(lambda x: np.log10(x + 1))
                mosi = mosi.rename(columns={immune_feat: f"-log10({immune_feat}+1)"})
                immune_feat = f"-log10({immune_feat}+1)"

            # Unique temperature states
            temperature_states = mosi['Mitotic Temperature'].unique()

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
            sns.boxplot(data=mosi, x='Mitotic Temperature', y=immune_feat, ax=ax, color=(1, 1, 1, 0.3), fliersize=0, linewidth=0.5, whis=0)
            sns.violinplot(data=mosi, x='Mitotic Temperature', y=immune_feat, ax=ax, palette=colors, fliersize=0, linewidth=0.5, alpha=0.5)


            # Perform pairwise statistical tests and add annotations
            pairs = list(itertools.combinations(temperature_states, 2))
            y_max = mosi[immune_feat].max()


            # # Perform pairwise statistical tests
            annotator = Annotator(ax, pairs, data=mosi, x='Mitotic Temperature', y=immune_feat, plot="violinplot")
            p_val_fmt = None # [[1e-4, "****"], [1e-3, "***"], [1e-2, "**"], [0.05, "*"], [1, "ns"]]
            annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', pvalue_format=p_val_fmt, hide_non_significant=False, verbose=2, alpha=0.01,
                                show_test_name=False)
            annotator.apply_and_annotate()

            ax.set_xlabel(None)
            if immune_feat == "Proliferation":
                ax.set_ylim([-3, 3])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Customize the plot
            plt.xticks(rotation=0)
            if cancer_type != "Pan-cancer":
                plt.title(cancer_type)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/immune_{immune_feat}_in_Hot-Cold.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
        except:
            print("does not work for : ", immune_feat)