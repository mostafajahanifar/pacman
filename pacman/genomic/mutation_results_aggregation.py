import os

import pandas as pd

from pacman.config import ALL_CANCERS, RESULTS_DIR

print(7*"="*7)
print("Aggregating Gene Mutation Results")
print(7*"="*7)

# Directory containing the cancer types' subdirectories
base_dir = f"{RESULTS_DIR}/genomic/mutation_pval"

# Create a writer to write results into an Excel file
output_file = f"{RESULTS_DIR}/genomic/mutation_pval/gene_mutation_aggregated.xlsx"
writer = pd.ExcelWriter(output_file, engine='openpyxl')

# Function to process each cancer type
def process_cancer_type(cancer_type, auc_file, pval_file, writer):
    # Read AUC and p-value matrices
    auc_matrix = pd.read_csv(auc_file, index_col=0)
    pval_matrix = pd.read_csv(pval_file, index_col=0)

    # Filter genes with at least one p-value < 0.05 (significant correlations)
    significant_genes = pval_matrix[pval_matrix < 0.05].dropna(how='all').index

    if len(significant_genes) >=5:
        auc_matrix = auc_matrix.loc[significant_genes]
        pval_matrix = pval_matrix.loc[significant_genes]

        # Sort genes based on the maximum correlation across features
        auc_matrix_rank = auc_matrix.copy()
        auc_matrix_rank[pval_matrix > 0.05] = 0
        auc_values = auc_matrix_rank.abs().max(axis=1)
        # keep the genes that have max corr higer than 0.2
        auc_values = auc_values[auc_values>0.2]
        sorted_genes = auc_values.sort_values(ascending=False).index

        # Reorder both AUC and p-value matrices
        auc_matrix = auc_matrix.loc[sorted_genes]
        pval_matrix = pval_matrix.loc[sorted_genes]
    else:
        auc_matrix_rank = auc_matrix.copy()
        auc_values = auc_matrix_rank.abs().max(axis=1)
        # keep the genes that have max corr higer than 0.2
        sorted_genes = auc_values.sort_values(ascending=False).index[:5]

        # Reorder both AUC and p-value matrices
        auc_matrix = auc_matrix.loc[sorted_genes]
        pval_matrix = pval_matrix.loc[sorted_genes]



    # Format AUC values and append "*" if corresponding p-value < 0.05
    formatted_auc_matrix = auc_matrix.copy()
    for row in formatted_auc_matrix.index:
        for col in formatted_auc_matrix.columns:
            auc_value = auc_matrix.at[row, col]
            pval_value = pval_matrix.at[row, col]
            formatted_value = f"{auc_value:.2f}"
            if pval_value < 0.05:
                formatted_value += "*"
            formatted_auc_matrix.at[row, col] = formatted_value

    # Write the formatted matrix to a sheet in the Excel file
    formatted_auc_matrix.to_excel(writer, sheet_name=cancer_type)

# Loop through each cancer type's directory
for cancer_type in ALL_CANCERS:
    cancer_dir = os.path.join(base_dir, cancer_type)
    
    if os.path.isdir(cancer_dir):
        auc_file = os.path.join(cancer_dir, f"{cancer_type}_auc_matrix.csv")
        pval_file = os.path.join(cancer_dir, f"{cancer_type}_pval_matrix.csv")
        
        if os.path.exists(auc_file) and os.path.exists(pval_file):
            print(f"Processing {cancer_type}")
            process_cancer_type(cancer_type, auc_file, pval_file, writer)

# Save the Excel file
# writer.save
writer.close()

print(f"Results aggregated and saved to {output_file}")
