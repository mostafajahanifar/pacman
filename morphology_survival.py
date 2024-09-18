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
from matplotlib.offsetbox import AnchoredText

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

from sklearn.cluster import AgglomerativeClustering
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.cluster import KMeans


save_root = "results_final/morphology/survival_WAF/"

ALL_CANCERS = ['BRCA', 'GBMLGG', 'COADREAD', 'KIRC', 'UCEC', 'LUSC', 'LUAD', 'HNSC',
       'THCA', 'SKCM', 'BLCA', 'STAD', 'LIHC', 'PRAD', 'KIRP', 'CESC', 'SARC']
# PFI
# ALL_CANCERS = ['GBMLGG', 'SKCM', 'LUAD', 'HNSC', 'LIHC', 'BLCA', 'KIRC', 'COADREAD',
#        'BRCA', 'LUSC', 'STAD', 'SARC', 'UCEC', 'PAAD', 'ESCA', 'OV', 'CESC',
#        'PRAD', 'KIRP', 'THCA', 'MESO', 'TGCT', 'UCS', 'ACC', 'CHOL']

# # #OS
# ALL_CANCERS = ['GBMLGG', 'HNSC', 'LUSC', 'SKCM', 'BLCA', 'KIRC', 'LUAD', 'BRCA',
#        'STAD', 'LIHC', 'COADREAD', 'PAAD', 'SARC', 'UCEC', 'OV', 'ESCA',
#        'CESC', 'MESO', 'KIRP', 'UCS', 'ACC', 'CHOL']

# #DSS
# ALL_CANCERS = ['GBMLGG', 'SKCM', 'HNSC', 'BLCA', 'KIRC', 'LUAD', 'STAD', 'LUSC',
#        'LIHC', 'BRCA', 'PAAD', 'COADREAD', 'SARC', 'OV', 'UCEC', 'CESC',
#        'ESCA', 'MESO', 'KIRP', 'UCS', 'ACC', 'CHOL']

# # DFI
# ALL_CANCERS = ['LIHC', 'LUAD', 'BRCA', 'LUSC', 'SARC', 'UCEC', 'STAD', 'BLCA',
#        'KIRP', 'TGCT', 'OV', 'CESC', 'PAAD', 'ESCA']

# keep only columns that are related to mutations
selected_feats = [
# "aty_hotspot_count",
# "aty_hotspot_ratio",
"aty_wsi_ratio",
]

mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final_ClusterByCancerNew_withAtypicalNew.csv')
# mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type", "temperature"]+selected_feats]
# mitosis_feats.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type", "temperature"] else col for col in mitosis_feats.columns]
mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")

for cancer_type in  ALL_CANCERS + ["All"]: # ALL_CANCERS +
    print(cancer_type)
    
    save_dir = f"{save_root}/{cancer_type}"
    os.makedirs(save_dir, exist_ok=True)

    censor_at = 120
    for event_type in ["OS", "PFI", "DSS", "DFI"]:
        event_time = f"{event_type}.time"

        for temp in ["Hot", "Cold"]:
            if cancer_type == "All":
                mitosis_feats_cancer = mitosis_feats#[mitosis_feats["type"].isin(ALL_CANCERS)]
            else:
                mitosis_feats_cancer = mitosis_feats[mitosis_feats["type"]==cancer_type]
            # keep only mitotic-hot cases
            mitosis_feats_cancer = mitosis_feats_cancer[mitosis_feats_cancer["temperature"]==temp]


            mosi = mitosis_feats_cancer[selected_feats+[event_type, event_time]]
            mosi = mosi.dropna(axis=0, how="any")

            # Reformat time events
            mosi = mosi.reset_index(drop=True)
            mosi[event_time] = mosi[event_time] / 30
            # Censor at 10 years
            ids = mosi[mosi[event_time] > censor_at].index
            mosi.loc[ids, event_time] = censor_at
            mosi.loc[ids, event_type] = 0


            feat_med = mosi[selected_feats].median().values[0]
            print(feat_med)
            if np.isnan(feat_med):
                print(f"Nan Med: {temp} -- {cancer_type}")
                continue
            mosi["labeled_cluster"] = mosi[selected_feats[0]].apply(lambda x: f"{temp} High" if x > feat_med else f"{temp} Low")
    
            
            # Create a color palette
            cluster_colors = {"Hot High": "brown",
                            "Hot Low": "orange",
                            "Cold High": "skyblue",
                            "Cold Low": "royalblue"}

            # Plot Kaplan-Meier survival plots
            kmf = KaplanMeierFitter()
            plt.figure(figsize=(2, 2))

            # Store Kaplan-Meier plots to add at risk counts later
            for cluster in mosi['labeled_cluster'].unique():
                mask = mosi['labeled_cluster'] == cluster
                kmf.fit(mosi.loc[mask, event_time], event_observed=mosi.loc[mask, event_type], label=cluster)
                ax = kmf.plot(ci_show=False, color=cluster_colors[cluster])


            # Perform pairwise log-rank tests and annotate p-values
            p_values = {}
            clusters = mosi['labeled_cluster'].unique()
            if len(clusters)<2:
                continue
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    cluster1 = clusters[i]
                    cluster2 = clusters[j]
                    mask1 = mosi['labeled_cluster'] == cluster1
                    mask2 = mosi['labeled_cluster'] == cluster2
                    results = logrank_test(mosi.loc[mask1, event_time], mosi.loc[mask2, event_time],
                                        event_observed_A=mosi.loc[mask1, event_type], event_observed_B=mosi.loc[mask2, event_type])
                    p_values[(cluster1, cluster2)] = results.p_value
            # Open the file in write mode
            with open(f"{save_dir}/km_{event_type}_{selected_feats}_censor{censor_at}.txt", 'w') as f:
                for cats, pval in p_values.items():
                    print(f"{cats[0]} vs {cats[1]}: p-value={pval:.02}", file=f)

            corrected_p_value = p_values[(cluster1, cluster2)]
            # adding the corrected p-value to the figure
            if corrected_p_value < 0.0001:
                pvalue_txt = 'p < 0.0001'
            else:
                pvalue_txt = 'p = ' + str(np.round(corrected_p_value, 4))
            pvalue_loc = 'upper right' if cancer_type=="GBMLGG" else 'lower left'
            ax.add_artist(AnchoredText(pvalue_txt, loc=pvalue_loc, frameon=False))
            # pval_text = ax.text(0.1, 0.05, pvalue_txt, transform=ax.transAxes)
            # adjust_text([pval_text])

            plt.xlim([0, 120])
            plt.xlabel('Time (Months)')
            plt.ylabel(f'{event_type} Probability')
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Position legend outside and to the right of the plot box
            plt.legend()
            ax.legend().set_visible(False)
            ax.set_title(cancer_type)
            ax.set_ylim([0, 1])
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{temp}_km_{event_type}_{selected_feats}_censor{censor_at}.png", dpi=600, bbox_inches = 'tight', pad_inches = 0.01)