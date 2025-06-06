import os

import pandas as pd

from pacman.config import ALL_CANCERS, DATA_DIR, RESULTS_DIR

print(7*"="*7)
print("Finding PACMAN set as highly correlated genes with mitosis features")
print("This should be run after correlation_measuring.py with mode=expression")
print(7*"="*7)

corr_thresh = 0.6
print(f"Using correlation threshold: {corr_thresh}")
# Directory where your subdirectories are located
base_directory = os.path.join(RESULTS_DIR, "gene/expression/")

# importing feature data to extract number of cases in each study
df = pd.read_excel(os.path.join(DATA_DIR, "ST1-tcga_mtfs.xlsx"))

# Initialize an empty set to store unique gene names
pacman_set = set([])

# Walk through each subdirectory to find PACMAN genes
for root, dirs, files in os.walk(base_directory):
    # Check if both corr_p.csv and corr_r.csv exist in the directory
    if 'corr_r.csv' in files and 'corr_p.csv' in files:
        # Get the subdirectory name
        subdirectory_name = os.path.basename(root)

        temp_df = df[df['type']==subdirectory_name]
        # Load correlation and p-value files
        corr_r_df = pd.read_csv(os.path.join(root, 'corr_r.csv'))
        corr_p_df = pd.read_csv(os.path.join(root, 'corr_p.csv'))

        corr_r_df=corr_r_df.set_index("Unnamed: 0")
        corr_p_df=corr_p_df.set_index("Unnamed: 0")

        # make the corr_r equal to 0 for the genes with non-significant p-value
        corr_r_df[corr_p_df>0.01] = 0

        corr_r_max = corr_r_df.abs().max(axis=0)
        selected_genes = corr_r_max[corr_r_max>corr_thresh].index

        pacman_set = pacman_set.union(set(selected_genes))

        print(subdirectory_name, len(selected_genes))


# Now, extract correlation measures for the PACMAN set
output_file = f"{base_directory}/pacman_genes_correlation_aggregated.xlsx"
writer = pd.ExcelWriter(output_file, engine='openpyxl')
gene_list = sorted(list(pacman_set))


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
for cancer_type in ["PAN-CANCER"] + ALL_CANCERS:
    cancer_dir = os.path.join(base_directory, cancer_type)

    if os.path.isdir(cancer_dir):
        print("Collecting correlations from: ", cancer_type)
        corr_file = os.path.join(cancer_dir, f"corr_r.csv")
        pval_file = os.path.join(cancer_dir, f"corr_p.csv")

        if os.path.exists(corr_file) and os.path.exists(pval_file):
            process_correlation_cancer_type(cancer_type, corr_file, pval_file, writer)

# Save the Excel file
writer.close()

print(f"Correlation results aggregated and formatted saved to {output_file}")
