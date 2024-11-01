import os
import pandas as pd
import numpy as np
from openpyxl import Workbook

# Directory containing the cancer types' subdirectories
base_dir = "results_final_all/gene/expression"
# Create a writer to write results into an Excel file
output_file = "results_final_all/gene/expression/pacman_genes_correlation_aggregated.xlsx"
writer = pd.ExcelWriter(output_file, engine='openpyxl')

gene_list = pd.read_csv("results_final_all/gene/expression/filtered_results_major_cancers.csv", header=None)
gene_list = sorted(gene_list[0].to_list())


# Function to process each cancer type for correlation analysis
def process_correlation_cancer_type(cancer_type, corr_file, pval_file, writer):
    # Read correlation and p-value matrices
    corr_matrix = pd.read_csv(corr_file, index_col=0)
    pval_matrix = pd.read_csv(pval_file, index_col=0)

    # Transpose to have genes as index and features as columns
    corr_matrix = corr_matrix.T
    pval_matrix = pval_matrix.T

    # select only genes from the list
    corr_matrix = corr_matrix.loc[gene_list]
    pval_matrix = pval_matrix.loc[gene_list]

    # # Filter genes with at least one p-value < 0.05 (significant correlations)
    # significant_genes = pval_matrix[pval_matrix < 0.05].dropna(how='all').index
    # if significant_genes.empty:
    #     print(f"No signicant gene for {cancer_type}")
    #     return -1
    # corr_matrix = corr_matrix.loc[significant_genes]
    # pval_matrix = pval_matrix.loc[significant_genes]

    # Sort genes based on the maximum correlation across features
    corr_r_matrix_rank = corr_matrix.copy()
    corr_r_matrix_rank[pval_matrix > 0.05] = 0
    max_corr_values = corr_r_matrix_rank.abs().max(axis=1)
    sorted_genes = max_corr_values.sort_values(ascending=False).index

    # Reorder the matrices based on sorted genes
    corr_matrix = corr_matrix.loc[sorted_genes]
    pval_matrix = pval_matrix.loc[sorted_genes]
    
    # Format the correlation values: 2 decimals and add '*' if p-value < 0.05
    formatted_corr_matrix = corr_matrix.copy()
    for gene in corr_matrix.index:
        for feature in corr_matrix.columns:
            corr_value = corr_matrix.at[gene, feature]
            pval_value = pval_matrix.at[gene, feature]
            formatted_value = f"{corr_value:.2f}"  # Format to 2 decimal places
            if pval_value < 0.05:
                formatted_value = f"{formatted_value}*"  # Add star for significant values
            formatted_corr_matrix.at[gene, feature] = formatted_value

    # Write the formatted correlation matrix to an Excel sheet
    formatted_corr_matrix.to_excel(writer, sheet_name=cancer_type)


# Loop through each cancer type's directory
for cancer_type in sorted(os.listdir(base_dir)):
    cancer_dir = os.path.join(base_dir, cancer_type)
    
    if os.path.isdir(cancer_dir):
        print(cancer_type)
        corr_file = os.path.join(cancer_dir, f"corr_r.csv")
        pval_file = os.path.join(cancer_dir, f"corr_p.csv")
        
        if os.path.exists(corr_file) and os.path.exists(pval_file):
            process_correlation_cancer_type(cancer_type, corr_file, pval_file, writer)

# Save the Excel file
writer.close()

print(f"Correlation results aggregated and formatted saved to {output_file}")
