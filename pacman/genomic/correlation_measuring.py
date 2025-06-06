import argparse
import os

import pandas as pd

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR
from pacman.utils import calculate_corr_matrix

selected_feats = [
    "HSC",
    "mean(ND)",
    "cv(ND)",
    "mean(CL)",
    "mean(HC)",
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Correlation Analysis between Gene expression and Methylation data")
    parser.add_argument("--mode", type=str, default="expression", choices=["mrna", "methylation", "expression"],
                        help="Mode of analysis: 'mrna' for mRNA, 'methylation' for methylation, 'expression' for both")
    parser.add_argument("--method", type=str, default="spearman", choices=["spearman", "pearson"],
                        help="Correlation method to use: 'spearman' or 'pearson'")
    args = parser.parse_args()

    print(7*"="*7)
    print(f"Running Correlation Analysis on f{args.mode} data")
    print(7*"="*7)

    #reading necessary data
    mitosis_feats = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))
    mitosis_feats = mitosis_feats[["bcr_patient_barcode", "type"]+selected_feats]

    if args.mode.lower() in ["mrna", "expression"]:
        gene_data_all = pd.read_csv(os.path.join(DATA_DIR, "tcga_all_gene_expressions_normalized.csv"))
    elif args.mode.lower() == "methylation":
        gene_data_all = pd.read_csv(os.path.join(DATA_DIR, "tcga_all_gene_methylation.csv"))
    else:
        raise ValueError("Invalid mode. Choose 'mrna', 'methylation', or 'expression'.")

    for cancer_types in ["Pan-cancer"]+ALL_CANCERS:
        print(cancer_types)
        cancer_types = [cancer_types]
        if cancer_types != ["Pan-cancer"]:
            print(f"Working on cancer types: {cancer_types}")
            mitosis_feats_cancer = mitosis_feats.loc[mitosis_feats['type'].isin(cancer_types)] # cancer types should be given in input argparse
        else:
            mitosis_feats_cancer = mitosis_feats.loc[mitosis_feats['type'].isin(ALL_CANCERS)]
            print("Working on ALL cancer types together")

        cancer_types_name = ''.join(cancer_types).upper()
        save_root = os.path.join(RESULTS_DIR, f"genomic/{args.mode}/{cancer_types_name}")
        os.makedirs(save_root, exist_ok=True)

        # Find the common case names between mitosis features and gene expressions
        common_cases = pd.Series(list(set(mitosis_feats_cancer['bcr_patient_barcode']).intersection(set(gene_data_all['case_id']))))
        ## Keep only the rows with the common case names in both dataframes
        df1_common = mitosis_feats_cancer[mitosis_feats_cancer['bcr_patient_barcode'].isin(common_cases)]
        df2_common = gene_data_all[gene_data_all['case_id'].isin(common_cases)]
        ## Sort the dataframes based on 'case_name'
        df1_common = df1_common.sort_values('bcr_patient_barcode')
        df2_common = df2_common.sort_values('case_id')
        ## Remove duplicate rows based on 'case_name' in df2_common
        df2_common = df2_common.drop_duplicates(subset='case_id')
        ## keep only feature and gene data
        X = df1_common.drop(columns=["bcr_patient_barcode", "type"])#.values
        Y = df2_common.drop(columns=['case_id', "type"])

        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

        if len(X)<10:
            print(f"Only {len(X)} pairs found, not enough to correlate...skipped")
            continue
        print(f"Correlating {len(X)} pairs...")

        corr_matrix, pvalue_matrix = calculate_corr_matrix(X, Y, method=args.method, pvalue_correction="fdr_bh")

        corr_matrix.to_csv(save_root+"corr_r.csv")
        pvalue_matrix.to_csv(save_root+"corr_p.csv")