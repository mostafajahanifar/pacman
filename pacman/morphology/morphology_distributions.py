import os, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pacman.utils import featre_to_tick, get_colors_dict
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
from sklearn.mixture import GaussianMixture

mit_temp = "All" # All, Cold, Hot
save_root = "results_final_all/morphology/distributions_AFW/"
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
    # "aty_ahotspot_count",
    "aty_wsi_ratio",
]

immune_feats = [
                # "SNV Neoantigens",
                # "Indel Neoantigens",
                # "Nonsilent Mutation Rate",
                # "CTA Score",
                "Number of Segments",
                "Fraction Altered",
                "Aneuploidy Score",
                "Homologous Recombination Defects",
                ]

mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancer_withAtypical.csv')
mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]+selected_feats]
mitosis_feats.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type", "temperature"] else col for col in mitosis_feats.columns]
mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")

for cancer_type in ['Pan-cancer']+ALL_CANCERS:
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

    if mit_temp in ["Hot", "Cold"]:
        mitosis_feats_cancer = mitosis_feats_cancer[mitosis_feats_cancer["temperature"]==mit_temp]


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

    feat_name = featre_to_tick(selected_feats[0])
    
    for immune_feat in immune_feats:
        try:
            mosi = pd.concat([df1_common[feat_name], df2_common[immune_feat]], axis=1)
            mosi = mosi.dropna(axis=0, how="any")

            # find cut_off to separate high and low values
            cutoff = np.median(mosi[feat_name].values.reshape(-1, 1))

            mosi[f'{feat_name} Temperature'] = mosi[feat_name].apply(lambda x: 'High' if x > cutoff else 'Low')

            mosi = mosi.sort_values(by=f'{feat_name} Temperature', ascending=False)

            #standardize the values
            if immune_feat in ["SNV Neoantigens", "Indel Neoantigens", "Nonsilent Mutation Rate", "Number of Segments"]:
                mosi[immune_feat] = mosi[immune_feat].apply(lambda x: np.log10(x+1))
                mosi = mosi.rename(columns={immune_feat: f"log10({immune_feat}+1)"})
                immune_feat = f"log10({immune_feat}+1)"

            # Unique temperature states
            temperature_states = mosi[f'{feat_name} Temperature'].unique()

            # Create the boxplot using matplotlib
            plt.figure(figsize=(1.8,3))
            ax = plt.gca()

            # Boxplot
            # Create a color palette
            cluster_colors = {f"Mitotic-Hot; {feat_name}-High": "brown",
                            f"Mitotic-Hot; {feat_name}-Low": "orange",
                            f"Mitotic-Cold; {feat_name}-High": "skyblue",
                            f"Mitotic-Cold; {feat_name}-Low": "royalblue",
                            f"Mitotic-All; {feat_name}-Low": "blue",
                            f"Mitotic-All; {feat_name}-High": "red",}
            colors = [cluster_colors[f"Mitotic-{mit_temp}; {feat_name}-Low"], cluster_colors[f"Mitotic-{mit_temp}; {feat_name}-High"]]
            sns.boxplot(data=mosi, x=f'{feat_name} Temperature', y=immune_feat, ax=ax, color=(1, 1, 1, 0.3), fliersize=0, linewidth=0.5, whis=0)
            sns.violinplot(data=mosi, x=f'{feat_name} Temperature', y=immune_feat, ax=ax, palette=colors, fliersize=0, linewidth=0.5, alpha=0.5)


            # Perform pairwise statistical tests and add annotations
            pairs = list(itertools.combinations(temperature_states, 2))
            y_max = mosi[immune_feat].max()


            # # Perform pairwise statistical tests
            annotator = Annotator(ax, pairs, data=mosi, x=f'{feat_name} Temperature', y=immune_feat, plot="violinplot")
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

            save_dir = save_root + cancer_type
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/morphology_mitotic{mit_temp}_{immune_feat}-in-{feat_name}_gmm.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)
        except Exception as e:
            print(f"does not work for : {immune_feat} because {e}")