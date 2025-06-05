import os, glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.model_selection import KFold, StratifiedKFold
from scipy import stats
from utils import featre_to_tick, get_colors_dict
import argparse
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

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

def plot_clustermap(corr_matrix, plvalue_matrix, mode, limit_row_cols=True, sig_threshold=0.05):
    # Set parameters based on the mode
    if mode == 'top':
        threshold = 0.2
        threshold_percentage_rows = 10# 30
        threshold_percentage_cols = 0# 10
        max_col_num = 13
        max_row_num = 25
        fig_size = (10, 10)
        dendo_ratio = (0.1, 0.1)
        cbar_pos= (0.02, 0.83, 0.02, 0.15)
        y_ticks = True
        astrisk = "*"
    elif mode == 'topTop':
        threshold = 0.2
        threshold_percentage_rows = 10# 45
        threshold_percentage_cols = 0# 15
        max_col_num = 8
        max_row_num = 10
        fig_size = (3, 3)
        dendo_ratio = (0.1, 0.1)
        cbar_pos= (0.75, 0.15, 0.02, 0.15)
        y_ticks = True
        astrisk = "*"
    elif mode == 'all':
        threshold = 0.05
        threshold_percentage_rows = 10
        threshold_percentage_cols = 0
        max_row_num = 100
        max_col_num = 30
        fig_size = (10, 13)
        dendo_ratio = (0.1, 0.07)
        cbar_pos= (0.02, 0.83, 0.02, 0.15)
        y_ticks = "auto"
        astrisk = "."
    else:
        raise ValueError("Invalid mode. Choose either 'top' or 'topTop'.")

    df_filtered = corr_matrix.copy()
    df_filtered[df_filtered.abs()<threshold] = np.nan
    df_filtered[plvalue_matrix>sig_threshold] = np.nan

    non_nan_counts_rows = df_filtered.count(axis=1)
    total_cells_rows = df_filtered.shape[1]
    percentage_non_nan_rows = (non_nan_counts_rows / total_cells_rows) * 100

    non_nan_counts_cols = df_filtered.count(axis=0)
    total_cells_cols = df_filtered.shape[0]
    percentage_non_nan_cols = (non_nan_counts_cols / total_cells_cols) * 100

    filtered_corr_matrix_rows = corr_matrix[percentage_non_nan_rows > threshold_percentage_rows]
    filtered_plvalue_matrix_rows = plvalue_matrix[percentage_non_nan_rows > threshold_percentage_rows]

    final_corr_matrix = filtered_corr_matrix_rows.loc[:, percentage_non_nan_cols > threshold_percentage_cols]
    final_plvalue_matrix = filtered_plvalue_matrix_rows.loc[:, percentage_non_nan_cols > threshold_percentage_cols]

    # Add the following lines to limit the number of rows and columns based on the sum of absolute correlation values
    if final_corr_matrix.shape[0] > max_row_num and limit_row_cols:
        row_sums = final_corr_matrix.abs().sum(axis=1)
        top_rows = row_sums.nlargest(max_row_num).index
        final_corr_matrix = final_corr_matrix.loc[top_rows]
        final_plvalue_matrix = final_plvalue_matrix.loc[top_rows]

    if final_corr_matrix.shape[1] > max_col_num and limit_row_cols:
        col_sums = final_corr_matrix.abs().sum(axis=0)
        top_cols = col_sums.nlargest(max_col_num).index
        final_corr_matrix = final_corr_matrix.loc[:, top_cols]
        final_plvalue_matrix = final_plvalue_matrix.loc[:, top_cols]

    significant = final_plvalue_matrix < sig_threshold

    g = sns.clustermap(final_corr_matrix, cmap="coolwarm",  method="ward", yticklabels=y_ticks, vmin=-0.75, vmax=0.75,
                       figsize=fig_size, dendrogram_ratio=dendo_ratio, cbar_pos=cbar_pos)

    row_labels = g.dendrogram_row.reordered_ind
    col_labels = g.dendrogram_col.reordered_ind

    for text in g.ax_heatmap.texts:
        text.set_visible(False)
    for i, row_label in enumerate(row_labels):
        for j, col_label in enumerate(col_labels):
            if significant.iloc[row_label, col_label]:
                g.ax_heatmap.text(j+0.5, i+0.5, astrisk, ha='center', va='center', color='black')

    return g.fig
   
def calculate_corr_matrix(df1, df2, method='spearman', pvalue_correction="fdr_bh"):
    if method not in ['spearman', 'pearson']:
        raise ValueError("Method must be 'spearman' or 'pearson'")

    corr_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=np.float32)
    pvalue_matrix = pd.DataFrame(index=df1.columns, columns=df2.columns, dtype=np.float32)
    for row in df1.columns:
        for col in df2.columns:
            if method == 'spearman':
                corr, pvalue = stats.spearmanr(df1[row], df2[col])
            elif method == 'pearson':
                corr, pvalue = stats.pearsonr(df1[row], df2[col])
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
    # if correct_pvalue_based_on=="df1":
    #     scale = len(df1.columns)
    # else:
    #     scale = len(df2.columns)
    # pvalue_matrix = pvalue_matrix * scale
    return corr_matrix, pvalue_matrix

def populate_results_dict(res_dict, cancer_type, gene_group, overall_pearsonr, overall_spearmanr, pears_corrs=None, spears_corrs=None):
    res_dict["type"].append(cancer_type)
    res_dict["gene_group"].append(gene_group)
    res_dict["overall_pearsonr"].append(overall_pearsonr)
    res_dict["overall_spearmanr"].append(overall_spearmanr)
    if pears_corrs!=None and spears_corrs!=None:
        res_dict["mean_pears_corrs"].append(np.mean(pears_corrs))
        res_dict["std_pears_corrs"].append(np.std(pears_corrs))
        res_dict["mean_spears_corrs"].append(np.mean(spears_corrs))
        res_dict["std_spears_corrs"].append(np.std(spears_corrs))
    return res_dict

def single_canonical(X, Y, method="PLSCanonical"):
    # Initialize CCA
    if method=="PLSCanonical":
        plsca = PLSCanonical(n_components=2)
    elif method=="CCA":
        plsca = CCA(n_components=2)
    else:
        raise ValueError("unknown method")
    
    X_r, Y_r = plsca.fit_transform(X, Y)

    # calculate overall correlations
    res = stats.pearsonr(X_r[:, 0], Y_r[:, 0])
    overall_pearsonr = res.statistic
    res = stats.spearmanr(X_r[:, 0], Y_r[:, 0])
    overall_spearmanr = res.statistic

    xrot = plsca.x_rotations_
    yrot = plsca.y_rotations_

    return overall_pearsonr, overall_spearmanr, xrot, yrot, X_r, Y_r 

def k_fold_canonical(X, Y, method="PLSCanonical", num_folds=5, stratify_by_type=None):
    # Initialize CCA or PLSCanonical
    if method == "PLSCanonical":
        plsca = PLSCanonical(n_components=2)
    elif method == "CCA":
        plsca = CCA(n_components=2)
    else:
        raise ValueError("Unknown method")

    # Initialize KFold or StratifiedKFold
    if stratify_by_type is not None:
        kf = StratifiedKFold(n_splits=num_folds)
        splits = kf.split(X, stratify_by_type)
    else:
        kf = KFold(n_splits=num_folds)
        splits = kf.split(X)

    # Initialize lists to store results
    X_r_vals = []
    Y_r_vals = []
    pears_corrs = []
    spears_corrs = []
    x_rots = []
    y_rots = []

    # Perform k-fold cross-validation
    for train_index, val_index in splits:
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        Y_train, Y_val = Y.iloc[train_index], Y.iloc[val_index]

        plsca.fit(X_train, Y_train)
        X_r_val, Y_r_val = plsca.transform(X_val, Y_val)

        res = stats.spearmanr(X_r_val[:, 0], Y_r_val[:, 0])
        spears_corrs.append(res.correlation)
        res = stats.pearsonr(X_r_val[:, 0], Y_r_val[:, 0])
        pears_corrs.append(res[0])

        X_r_vals.append(X_r_val)
        Y_r_vals.append(Y_r_val)

        x_rots.append(plsca.x_rotations_)
        y_rots.append(plsca.y_rotations_)

    # Concatenate results
    X_r_vals = np.concatenate(X_r_vals, axis=0)
    Y_r_vals = np.concatenate(Y_r_vals, axis=0)

    # Calculate overall correlations
    res = stats.pearsonr(X_r_vals[:, 0], Y_r_vals[:, 0])
    overall_pearsonr = res[0]
    res = stats.spearmanr(X_r_vals[:, 0], Y_r_vals[:, 0])
    overall_spearmanr = res.correlation

    avr_x_rot = np.mean(np.stack(x_rots, axis=2), axis=2)
    avr_y_rot = np.mean(np.stack(y_rots, axis=2), axis=2)

    return overall_pearsonr, overall_spearmanr, pears_corrs, spears_corrs, avr_x_rot, avr_y_rot, X_r_vals, Y_r_vals

def plot_comp_corr(X_r_train, Y_r_train, title, color_list=None):
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(X_r_train[:, 0], Y_r_train[:, 0], marker="o", s=5, alpha=0.2, color=color_list)
    plt.xlabel("SNA features canonical component 1")
    plt.ylabel("Gene expresions canonical component 1")
    plt.title(title)
    plt.xticks(())
    plt.yticks(())
    return fig

def plot_rot_map(xrot, yrot, x_names, y_names):
    def decide_alignment(x, y):
        if x<0 and y<0:
            h, v = "right", "top"
        if x<0 and y>0:
            h, v = "right", "bottom"
        if x>0 and y>0:
            h, v = "left", "bottom"
        if x>0 and y<0:
            h, v = "left", "top"
        return h,v
    fig, ax = plt.subplots(figsize=(4, 4)) 
    # plt.xlim((-1.01,1.01))
    # plt.ylim((-1.01,1.01))
    plt.xlim((-0.76,0.76))
    plt.ylim((-0.76,0.76))
    
    # first draw circles
    circle1 = plt.Circle((0, 0), 0.1, fill=False, color='black', alpha=0.5)
    circle2 = plt.Circle((0, 0), 0.5, fill=False, color='black', alpha=0.5)
    circle3 = plt.Circle((0, 0), 1.0, fill=False, color='black', alpha=0.5)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    # ax.add_patch(circle3)
    plt.axhline(0, color='black', alpha=0.25)
    plt.axvline(0, color='black', alpha=0.25)
    viz_thresh = 0.2
    for vi, var in enumerate(xrot):
        var = np.clip(var, -0.7, 0.7)
        # print(var)
        plt.arrow(0,0,var[0],var[1], color='red', alpha=0.1, head_width=0)
        plt.scatter(var[0],var[1], color='red', alpha=0.4,)
        h, v = decide_alignment(var[0], var[1])
        if abs(var[0])>viz_thresh or abs(var[1])>viz_thresh:
            plt.text(var[0],var[1],x_names[vi], color='red', horizontalalignment=h, verticalalignment=v)

    for vi, var in enumerate(yrot):
        plt.arrow(0,0,var[0],var[1], color='blue', alpha=0.1, head_width=0)
        plt.scatter(var[0],var[1], color='blue', alpha=0.4,)
        h, v = decide_alignment(var[0], var[1])
        if abs(var[0])>viz_thresh or abs(var[1])>viz_thresh:
            plt.text(var[0],var[1],y_names[vi], color='blue', horizontalalignment=h, verticalalignment=v)

    ax.set_xlabel("CCA Dimension 1")
    ax.set_ylabel("CCA Dimension 2")
    ax.set_aspect('equal')
    return fig

selected_feats = [
"mean(ND)",
"cv(ND)",
"mit_nodeDegrees_per99",
"mean(CL)",
"mit_clusterCoff_std",
"mit_clusterCoff_per90",
"mean(HC)",
"mit_cenHarmonic_std",
"mit_cenHarmonic_per10",
"mit_cenHarmonic_per99",
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gene to mitosis features analysis')
    parser.add_argument('--cancer_types', nargs='+', required=True)
    args = parser.parse_args()
    cancer_types = args.cancer_types

    #reading necessary data
    # mitosis_feats = pd.read_csv('/home/u2070124/lsf_workspace/Data/Data/pancancer/tcga_features_final.csv')
    mitosis_feats = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final.csv')
    signatures = pd.read_csv("gene/data/signatures.csv")
    gene_expr_all = pd.read_csv("gene/data/tcga_all_gene_expressions_normalized.csv")
    canonical_save_root = "results_final/gene/canonical_corr/"
    bicluster_save_root = "results_final/gene/bicluster_cross_corr/"
    func_corr_save_root = "results_final/gene/func_heatmap_corr/"
    cancer_colors = get_colors_dict()

    if cancer_types != ["all"]:
        print(f"Working on cancer types: {cancer_types}")
        mitosis_feats = mitosis_feats.loc[mitosis_feats['type'].isin(cancer_types)] # cancer types should be given in input argparse
    else:
        mitosis_feats = mitosis_feats.loc[mitosis_feats['type'].isin(ALL_CANCERS)]
        print("Working on ALL cancer types together")
    mitosis_feats["type"] = mitosis_feats["type"].replace(["COAD", "READ"], "COADREAD")
    mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type"]+selected_feats]
    mitosis_feats.columns = [featre_to_tick(col) if col not in ["bcr_patient_barcode", "type"] else col for col in mitosis_feats.columns]
    cancer_types_name = ''.join(cancer_types).upper()
    gene_groups = signatures.columns.tolist() # ["Mitosis","Mitosis Process", "Tumor Suppressor","Oncogenes","Protein Kinases"] # 

    # results place holders
    CCA_res_dict = {'type': [], "gene_group": [], "overall_pearsonr": [], "overall_spearmanr": [], "mean_pears_corrs": [], "std_pears_corrs": [], "mean_spears_corrs": [], "std_spears_corrs": []}
    PLS_res_dict = {'type': [], "gene_group": [], "overall_pearsonr": [], "overall_spearmanr": [], "mean_pears_corrs": [], "std_pears_corrs": [], "mean_spears_corrs": [], "std_spears_corrs": []}
    single_PLS_res_dict = {'type': [], "gene_group": [], "overall_pearsonr": [], "overall_spearmanr": []}
    single_CCA_res_dict = {'type': [], "gene_group": [], "overall_pearsonr": [], "overall_spearmanr": []}

    for gene_group in gene_groups:
        print(f"Started working on {gene_group}")
        # getting the gene list and filtering gene expression data
        gene_list = signatures[gene_group].dropna().to_list()
        # gene_list = [gene+'_rnaseq' for gene in gene_list]
        gene_list = [gene for gene in gene_list if gene in gene_expr_all.columns]
        gene_expr = gene_expr_all[["case_id"]+gene_list]

        # Find the common case names between mitosis features and gene expressions
        common_cases = pd.Series(list(set(mitosis_feats['bcr_patient_barcode']).intersection(set(gene_expr['case_id']))))
        ## Keep only the rows with the common case names in both dataframes
        df1_common = mitosis_feats[mitosis_feats['bcr_patient_barcode'].isin(common_cases)]
        df2_common = gene_expr[gene_expr['case_id'].isin(common_cases)]
        # ## remove the _rnaseq tail from the name of the 
        # df2_common.columns = [col.strip('_rnaseq') if col != 'case_id' else col for col in df2_common.columns]
        ## Sort the dataframes based on 'case_name'
        df1_common = df1_common.sort_values('bcr_patient_barcode')
        df2_common = df2_common.sort_values('case_id')
        ## Remove duplicate rows based on 'case_name' in df2_common
        df2_common = df2_common.drop_duplicates(subset='case_id')
        ## find the case type for stratification in the pan-cancer scenario
        stratify_by_type = df1_common['type'].to_list() if cancer_types_name=="ALL" else None
        ## find the color of points in pan-cancer scenario
        color_list = [cancer_colors[cancer] for cancer in df1_common["type"]] if cancer_types_name=="ALL" else None
        ## keep only feature and gene data
        X = df1_common.drop(columns=["bcr_patient_barcode", "type"])#.values
        Y = df2_common.drop(columns='case_id')#.values
        ## drop the gene column if the is any nan in it
        Y = Y.dropna(axis=1, how="any")

        # Do K-fold cross-validation canonical correlation analysis
        canon_save_path = os.path.join(canonical_save_root, cancer_types_name, gene_group)
        os.makedirs(canon_save_path, exist_ok=True)
        # overall_pearsonr, overall_spearmanr, pears_corrs, spears_corrs, pls_x_rot, pls_y_rot, X_r_vals, Y_r_vals = k_fold_canonical(X, Y, method="PLSCanonical", num_folds=5)
        # PLS_res_dict = populate_results_dict(PLS_res_dict, cancer_types_name, gene_group, overall_pearsonr, overall_spearmanr, pears_corrs, spears_corrs)
        # fig = plot_comp_corr(X_r_vals, Y_r_vals, f"PLSC-{gene_group}-Pearson corr.: {np.mean(pears_corrs):.2f}±{np.std(pears_corrs):.2f}")
        # fig.savefig(os.path.join(canon_save_path, f"cv_pls_pearson_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        # fig = plot_comp_corr(X_r_vals, Y_r_vals, f"PLSC-{gene_group} - Spearman corr.: {overall_spearmanr:.2f}")
        # fig.savefig(os.path.join(canon_save_path, f"cv_pls_spearman_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        # fig = plot_rot_map(pls_x_rot, pls_y_rot, df1_common.columns[1:], df2_common.columns[1:])
        # fig.savefig(os.path.join(canon_save_path, f"cv_pls_rotplot_{cancer_types_name}_{gene_group}.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0)

        overall_pearsonr, overall_spearmanr, pears_corrs, spears_corrs, cca_x_rot, cca_y_rot, X_r_vals, Y_r_vals = k_fold_canonical(X, Y, method="CCA", num_folds=5, stratify_by_type=stratify_by_type)
        CCA_res_dict = populate_results_dict(CCA_res_dict, cancer_types_name, gene_group, overall_pearsonr, overall_spearmanr, pears_corrs, spears_corrs)
        fig = plot_comp_corr(X_r_vals, Y_r_vals, f"{gene_group} (r={np.mean(pears_corrs):.2f}±{np.std(pears_corrs):.2f})", color_list)
        # add legend in case of pan-cancer scenario
        if cancer_types_name == "ALL":
            legend_elements = [mpatches.Patch(color=color, label=domain) for domain, color in cancer_colors.items()]
            fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.9, 0.92), ncol=2)
        fig.savefig(os.path.join(canon_save_path, f"cv_cca_pearson_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        # fig = plot_comp_corr(X_r_vals, Y_r_vals, f"CCA-{gene_group}-Spearman corr.: {overall_spearmanr:.2f}")
        # fig.savefig(os.path.join(canon_save_path, f"cv_cca_spearman_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        fig = plot_rot_map(cca_x_rot, cca_y_rot, df1_common.columns[1:], df2_common.columns[1:])
        fig.savefig(os.path.join(canon_save_path, f"cv_cca_rotplot_{cancer_types_name}_{gene_group}.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        fig.savefig(os.path.join(canon_save_path, f"cv_cca_rotplot_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        
        overall_pearsonr, overall_spearmanr, xrot, yrot, X_r, Y_r = single_canonical(X, Y, method="CCA")
        single_CCA_res_dict = populate_results_dict(single_CCA_res_dict, cancer_types_name, gene_group, overall_pearsonr, overall_spearmanr)
        fig = plot_comp_corr(X_r, Y_r, f"CCA-{gene_group}-Pearson corr.: {overall_pearsonr:.2f}")
        fig.savefig(os.path.join(canon_save_path, f"single_cca_pearson_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        # fig = plot_comp_corr(X_r, Y_r, f"CCA-{gene_group} - Spearman corr.: {overall_spearmanr:.2f}")
        # fig.savefig(os.path.join(canon_save_path, f"single_cca_spearman_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        fig = plot_rot_map(xrot, yrot, df1_common.columns[1:], df2_common.columns[1:])
        fig.savefig(os.path.join(canon_save_path, f"single_cca_rotplot_{cancer_types_name}_{gene_group}.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0)

        # # Normalization
        # X_norm = (X-X.mean()) / X.std()
        # Y_norm = (Y-Y.mean()) / Y.std()

        # # remove the columns with nans (zero standard deviation)
        # X_norm.dropna(axis=1, how="any", inplace=True)
        # Y_norm.dropna(axis=1, how="any", inplace=True)
        

        # # biclustering of all correlations
        # bicluster_save_path = os.path.join(bicluster_save_root, cancer_types_name, gene_group)
        # os.makedirs(bicluster_save_path, exist_ok=True)

        # try:
        #     corr_matrix, p_matrix = calculate_corr_matrix(Y_norm, X_norm, method='pearson', pvalue_correction="fdr_bh")
        #     fig = plot_clustermap(corr_matrix, p_matrix, mode="all",limit_row_cols=False)
        #     fig.savefig(os.path.join(bicluster_save_path, f"all_bicluster_pearson_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        #     fig.savefig(os.path.join(bicluster_save_path, f"all_bicluster_pearson_{cancer_types_name}_{gene_group}.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        # except:
        #     print("There is no even small correlation in this case")

        # # bilcustering of top features
        # try:
        #     fig = plot_clustermap(corr_matrix, p_matrix, mode="top")
        #     fig.savefig(os.path.join(bicluster_save_path, f"top_bicluster_pearson_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        #     fig.savefig(os.path.join(bicluster_save_path, f"top_bicluster_pearson_{cancer_types_name}_{gene_group}.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0)

        #     # bilcustering of topTop features
        #     fig = plot_clustermap(corr_matrix, p_matrix, mode="topTop")
        #     fig.savefig(os.path.join(bicluster_save_path, f"topTop_bicluster_pearson_{cancer_types_name}_{gene_group}.png"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        #     fig.savefig(os.path.join(bicluster_save_path, f"topTop_bicluster_pearson_{cancer_types_name}_{gene_group}.pdf"), dpi=600, bbox_inches = 'tight', pad_inches = 0)
        # except:
        #     print("Either top or topTop is not achieveable")
        # plt.close()

    # save dataframes to csv files
    CCA_res_df = pd.DataFrame(CCA_res_dict)
    # PLS_res_df = pd.DataFrame(PLS_res_dict)
    # single_PLS_res_df = pd.DataFrame(single_PLS_res_dict)
    single_CCA_res_df = pd.DataFrame(single_CCA_res_dict)
    cca_save_dir = os.path.join(canonical_save_root, cancer_types_name)
    CCA_res_df.to_csv(cca_save_dir + "/cv_CCA_res.csv", index=None)
    single_CCA_res_df.to_csv(cca_save_dir + "/single_CCA_res.csv", index=None)
    # PLS_res_df.to_csv(cca_save_dir + "/cv_PLS_res.csv", index=None)
    # single_PLS_res_df.to_csv(cca_save_dir + "/single_PLS_res.csv", index=None)