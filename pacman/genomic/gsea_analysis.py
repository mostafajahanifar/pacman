import argparse
import os

import gseapy as gp
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text
from gseapy import dotplot, enrichment_map
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from sanbomics.plots import volcano

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR

print(7*"="*7)
print(f"Running GSEA Analysis")
print(7*"="*7)

# setting parameters
log2fc_thresh = 1
pval_thresh = 1e-3
num_ann = 10

parser = argparse.ArgumentParser()
parser.add_argument("--cancer", type=str, help="Specify the cancer type")
args = parser.parse_args()

# Reading the data
mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))
mit_signatures = pd.read_csv(os.path.join(DATA_DIR, "signatures.csv"))["Mitosis Process"].dropna().to_list()
print("Reading data is done")

items = ALL_CANCERS if args.cancer.lower() == "all" else [args.cancer]
for cancer_type in items:
    counts_org = pd.read_csv(os.path.join(DATA_DIR,f'raw_gene_counts/{cancer_type}_gene_raw_counts.csv'))
    save_root = f"{RESULTS_DIR}/genomic/dseq_results/{cancer_type}/"
    os.makedirs(save_root, exist_ok=True)
    try:
        # filtering by cancer_type
        print(f"Started working on {cancer_type}")
        cancer_mitosis_feats = mitosis_feats[mitosis_feats["type"]==cancer_type]
        ## Find the common case names between mitosis features and gene expressions
        common_cases = pd.Series(list(set(cancer_mitosis_feats['bcr_patient_barcode']).intersection(set(counts_org['Case ID']))))
        ## Keep only the rows with the common case names in both dataframes
        df1_common = cancer_mitosis_feats[cancer_mitosis_feats['bcr_patient_barcode'].isin(common_cases)]
        df2_common = counts_org[counts_org['Case ID'].isin(common_cases)]
        ## Sort the dataframes based on 'case_name'
        df1_common = df1_common.sort_values('bcr_patient_barcode')
        df2_common = df2_common.sort_values('Case ID')
        ## Remove duplicate rows based on 'case_name' in df2_common
        df2_common = df2_common.drop_duplicates(subset='Case ID')

        # Extracting metadata df
        metadata = df1_common[['bcr_patient_barcode', 'temperature']]
        metadata = metadata.rename(columns={"bcr_patient_barcode": "Sample", "temperature": "Condition"})
        metadata = metadata.set_index('Sample')
        # Extracting counts df
        counts = df2_common.drop(columns=["Project ID"])
        counts= counts.rename(columns={"Case ID": "Geneid"})
        counts = counts.set_index("Geneid")
        counts = counts[counts.columns].astype(int)
        # get rid of redundant variables
        del df1_common, df2_common, common_cases

        # filtering the genes that have few counts
        counts = counts.loc[:, counts.sum() >= 100]

        # create DESeq dataset and fit
        print(f"Started DESeq ... {cancer_type}")
        dds = DeseqDataSet(counts=counts,
                    metadata=metadata,
                    design_factors="Condition")
        dds.deseq2()

        # Get DESeq stats
        stat_res = DeseqStats(dds, contrast = ('Condition','Hot','Cold'), alpha=0.01)
        stat_res.summary()

        # Extract signals from the results
        res = stat_res.results_df
        res["Symbol"] = res.index
        res = res[res.baseMean >= 10] # keep the results with high baseMean
        sigs = res[(res.padj < 0.05) & (abs(res.log2FoldChange) > 0.5)]
        sigs.to_csv(save_root+"deseq_out.csv", index=None)


        # GSEA analysis
        print(f"Performing GSEA ... {cancer_type}")
        ranking = res[['stat']].dropna().sort_values('stat', ascending = False)
        pre_res = gp.prerank(rnk = ranking,
                            gene_sets = ['Reactome_2022', 'KEGG_2021_Human', 'MSigDB_Hallmark_2020'],
                            threads=16,
                            seed = 6,
                            permutation_num = 10000,
                            outdir=save_root,
                            graph_num=20,
                            )
        ## plotting top 5 pathways together
        terms = pre_res.res2d.Term
        axs = pre_res.plot(terms=terms[:5],
                        show_ranking=True, # whether to show the second yaxis
                        figsize=(3,4),
                        ofname=save_root+"top5_pathways"
                        )


        # Volcano plot
        print(f"Volcano plot ... {cancer_type}")
        ## getting the list of genes to annotate
        mit_genes = [sym for sym in sigs["Symbol"] if sym in mit_signatures]
        shape_dict = {"Mitosis-related": mit_genes}
        sigs['sorter'] = -np.log10(sigs["padj"])*sigs["log2FoldChange"]

        df_sorted = sigs[(sigs["padj"]<pval_thresh) & (sigs["log2FoldChange"].abs()>log2fc_thresh)]
        df_sorted = df_sorted.sort_values(by='sorter', ascending=False)

        df_to_save = pd.concat([df_sorted.head(20), df_sorted.tail(20)], axis=0, ignore_index=True)
        df_to_save.to_csv(save_root+"top_df_genes.csv", index=None)

        top_rows = df_sorted["Symbol"].head(num_ann).to_list()
        bottom_rows = df_sorted["Symbol"].tail(num_ann).to_list()
        top_rows = top_rows+bottom_rows

        mitosis_rows = df_sorted[df_sorted['Symbol'].isin(mit_genes)]
        if len(mitosis_rows)>0:
            if len(mitosis_rows)>6:
                top_m_rows = mitosis_rows[mitosis_rows["sorter"]>0]["Symbol"].head(3).to_list()
                bottom_m_rows = mitosis_rows[mitosis_rows["sorter"]<0]["Symbol"].head(3).to_list()
                top_m_rows = top_m_rows + bottom_m_rows
                top_rows = list(set(top_rows).union(set(top_m_rows)))

        ## make and save the plot
        volcano(sigs, symbol='Symbol', baseMean="baseMean",
                log2fc_thresh=log2fc_thresh, pval_thresh=pval_thresh,
                to_label=top_rows, figsize=(5.5,5),
                colors=['skyblue', 'gainsboro', 'dodgerblue'],
                shape_dict=shape_dict,  fontsize=9,
                save=save_root+"volcano")

        # Dotplot of pathways
        print(f"Plotting pathways ... {cancer_type}")
        ax = dotplot(pre_res.res2d,
                    column="FDR q-val",
                    title=None,
                    cmap=plt.cm.coolwarm,
                    size=3, # adjust dot size
                    top_term=20,
                    figsize=(4,8), cutoff=0.25, show_ring=False,
                    ofname=save_root+"pathways_dotplot",
                    vmin=2,
                    vmax=5,)
    except Exception as e:
        print(100*"*")
        print(f"Something went wrong in {cancer_type}: {e}")
        print(100*"*")