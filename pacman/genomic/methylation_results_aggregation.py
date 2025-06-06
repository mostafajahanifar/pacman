import os

import pandas as pd

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR

print(7*"="*7)
print(f"Running Gene Methylation Results Aggregation")
print(7*"="*7)


def load_gene_name_mapping(mapping_file):
    """Add CpG id to all names"""
    df = pd.read_csv(mapping_file, sep="\t")
    df['NAME'] = df.apply(lambda row: row['ENTITY_STABLE_ID'] if pd.isna(row['NAME']) else row['NAME'], axis=1)

    # Append ENTITY_STABLE_ID to each name
    # df['NAME'] = df.apply(lambda row: f"{row['NAME']}_{row['ENTITY_STABLE_ID']}", axis=1)
    df['NAME'] = df.apply(lambda row: f"{row['NAME']}", axis=1)

    return df.set_index('ENTITY_STABLE_ID')['NAME'].to_dict()

# Directory containing the cancer types' subdirectories
base_dir = f"{RESULTS_DIR}/genomic/methylation"
# Create a writer to write results into an Excel file
output_file = f"{RESULTS_DIR}/genomic/methylation/gene_methylation_aggregated.xlsx"
writer = pd.ExcelWriter(output_file, engine='openpyxl')

mapping_file = f"{DATA_DIR}/data_methylation.txt"

# Load the gene ID to name mapping
id_to_name = load_gene_name_mapping(mapping_file)

# Function to process each cancer type for correlation analysis
def process_correlation_cancer_type(cancer_type, corr_file, pval_file, writer, id_to_name):
    # Read correlation and p-value matrices
    corr_matrix = pd.read_csv(corr_file, index_col=0)
    pval_matrix = pd.read_csv(pval_file, index_col=0)

    # Transpose to have genes as index and features as columns
    corr_matrix = corr_matrix.T
    pval_matrix = pval_matrix.T

    # Filter genes with at least one p-value < 0.05 (significant correlations)
    significant_genes = pval_matrix[pval_matrix < 0.05].dropna(how='all').index
    if significant_genes.empty:
        print(f"No signicant gene for {cancer_type}")
        return -1

    if len(significant_genes) >=5:
        corr_matrix = corr_matrix.loc[significant_genes]
        pval_matrix = pval_matrix.loc[significant_genes]

        # Sort genes based on the maximum correlation across features
        corr_r_matrix_rank = corr_matrix.copy()
        corr_r_matrix_rank[pval_matrix > 0.05] = 0
        max_corr_values = corr_r_matrix_rank.abs().max(axis=1)
        # keep the genes that have max corr higer than 0.2
        max_corr_values = max_corr_values[max_corr_values>0.2]
        sorted_genes = max_corr_values.sort_values(ascending=False).index
    else:
        # Sort genes based on the maximum correlation across features
        corr_r_matrix_rank = corr_matrix.copy()
        max_corr_values = corr_r_matrix_rank.abs().max(axis=1)
        sorted_genes = max_corr_values.sort_values(ascending=False).index[:5]

    # Reorder the matrices based on sorted genes
    corr_matrix = corr_matrix.loc[sorted_genes]
    pval_matrix = pval_matrix.loc[sorted_genes]

    corr_matrix["NAME"] = [id_to_name.get(gene, gene) for gene in corr_matrix.index]
    pval_matrix["NAME"] = corr_matrix["NAME"]
    corr_matrix = corr_matrix[["NAME"]+list(corr_matrix.columns[:-1])]
    pval_matrix = pval_matrix[["NAME"]+list(pval_matrix.columns[:-1])]

    corr_matrix=corr_matrix.rename(index={"": "ID"})
    pval_matrix=pval_matrix.rename(index={"": "ID"})

    # Format the correlation values: 2 decimals and add '*' if p-value < 0.05
    formatted_corr_matrix = corr_matrix.copy()
    for gene in corr_matrix.index:
        for feature in corr_matrix.columns[1:]:
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
            process_correlation_cancer_type(cancer_type, corr_file, pval_file, writer, id_to_name)

# Save the Excel file
writer.close()

print(f"Correlation results aggregated and formatted saved to {output_file}")
