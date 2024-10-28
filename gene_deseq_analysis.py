import os
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gseapy as gp
from gseapy.plot import gseaplot
import numpy as np
import seaborn as sns
from sanbomics.plots import volcano
from gseapy import dotplot
from gseapy import enrichment_map
import networkx as nx
from adjustText import adjust_text
import matplotlib.patheffects as pe
import matplotlib.cm as cm
import argparse

# setting parameters
log2fc_thresh = 1
pval_thresh = 1e-3
num_ann = 10

parser = argparse.ArgumentParser()
parser.add_argument("--cancer", type=str, help="Specify the cancer type")
args = parser.parse_args()
cancer_type = args.cancer

# Reading the data
mitosis_feats = pd.read_csv('/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_final_ClusterByCancerNew.csv')
mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
mitosis_feats["type"] = mitosis_feats["type"].replace(["GBM", "LGG"], "GBMLGG")

mit_signatures = pd.read_csv("gene/data/signatures.csv")["Mitosis Process"].dropna().to_list()
print("Reading data is done")

# for cancer_type in mitosis_feats["type"].unique():
counts_org = pd.read_csv(f'gene/data/raw_gene_counts/{cancer_type}_gene_raw_counts.csv')
save_root = f"results_final_all/gene/dseq_results/{cancer_type}/"
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

    # Plotting 2D and 3D PCA
    print(f"Plotting PCAs ... {cancer_type}")
    sc.tl.pca(dds)
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    fig = sc.pl.pca(dds, color = 'Condition', size = 30, alpha=0.3, ax=ax, show=False)
    plt.savefig(save_root+"pca2d.pdf", dpi=300, bbox_inches = 'tight', pad_inches = 0)

    sc.tl.pca(dds, n_comps=3)  # Specify the number of principal components to compute
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
    sc.pl.pca(dds, color='Condition', size=300, alpha=0.5, ax=ax, show=False, components='all', projection='3d')
    plt.savefig(save_root+"pca3d.pdf", dpi=300, bbox_inches='tight', pad_inches=0)

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

    # Netowork of top pathways
    top_term_num = 10
    nodes, edges = enrichment_map(pre_res.res2d, top_term=top_term_num)
    G = nx.Graph()
    ## Add nodes to the graph
    for idx, row in nodes.iterrows():
        G.add_node(idx, **row.to_dict())
    ## Add edges to the graph
    for idx, row in edges.iterrows():
        G.add_edge(row['src_idx'], row['targ_idx'], **row[['jaccard_coef', 'overlap_coef', 'overlap_genes']].to_dict())
    fig, ax = plt.subplots(figsize=(6, 5))
    ## init node cooridnates
    pos=nx.layout.kamada_kawai_layout(G)
    ## draw node
    cmap = plt.cm.coolwarm
    node_colors = list(nodes.NES)
    vmin=-3
    vmax=3
    nx.draw_networkx_nodes(G,
                        pos=pos,
                        cmap=cmap,
                        node_color=node_colors,
                        node_size=list(nodes.Hits_ratio *1000),
                        vmin=vmin, vmax=vmax,
                        )
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)), ax=ax,
                        fraction=0.02, pad=0.02
                        )
    cbar.ax.set_aspect("auto")  # Increase this number to reduce the width
    cbar.set_label('NES')
    cbar.outline.set_visible(False)
    ## draw node label
    texts = []
    for node, (x, y) in pos.items():
        texts.append(ax.text(x, y, s=nodes.loc[node, 'Term'], 
                            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                            horizontalalignment='center', 
                            verticalalignment='center'))
    ## Use adjust_text to prevent overlap
    adjust_text(texts) 
    ## draw edge
    edge_weight = nx.get_edge_attributes(G, 'jaccard_coef').values()
    nx.draw_networkx_edges(G,
                        pos=pos,
                        width=list(map(lambda x: x*top_term_num, edge_weight)),
                        edge_color='#CDDBD4')
    plt.savefig(save_root+"pathways_network.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
except Exception as e:
    print(100*"*")
    print(f"Something went wrong in {cancer_type}: {e}")
    print(100*"*")