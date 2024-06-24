import os
import pandas as pd
from tqdm import tqdm

# Paths to the link table, gene expression files, and HUGO gene symbol data
link_table_path = "/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_rna_raw_count.tsv"
expression_files_dir = "/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_rna_raw_count"
hugo_table_path = "gene/data/data_mrna_seq_v2_rsem_zscores_ref_all_samples.txt"
cache_dir = "/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/cache"

# Create cache directory if it doesn't exist
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Read the link table
link_df = pd.read_csv(link_table_path, sep="\t")

# Read the HUGO gene symbol data
hugo_table = pd.read_csv(hugo_table_path, sep="\t")
hugo_symbols = set(hugo_table["Hugo_Symbol"])

# Recursively find all .tsv files in the directory
expression_files = []
for root, dirs, files in os.walk(expression_files_dir):
    for file in files:
        if file.endswith('.tsv'):
            expression_files.append(os.path.join(root, file))

# Batch processing parameters
batch_size = 500
num_batches = (len(expression_files) + batch_size - 1) // batch_size

# Process files in batches
for batch_num in tqdm(range(num_batches)):
    batch_files = expression_files[batch_num * batch_size : (batch_num + 1) * batch_size]
    
    # Initialize an empty list to store individual DataFrames
    dfs = []
    
    for file_path in batch_files:
        # Extract the file name from the path
        file_name = os.path.basename(file_path)
        
        # Read the gene expression data
        df = pd.read_csv(file_path, sep="\t", skiprows=[0, 2, 3, 4, 5])
        
        # Filter genes based on HUGO symbols
        df = df[df["gene_name"].isin(hugo_symbols)]
        
        # Get the corresponding Case ID and Project ID
        case_id = link_df.loc[link_df['File Name'] == file_name, 'Case ID'].values[0]
        project_id = link_df.loc[link_df['File Name'] == file_name, 'Project ID'].values[0]
        
        # Add Case ID and Project ID to the DataFrame
        df['Case ID'] = case_id
        df['Project ID'] = project_id
        
        # Select and rename the relevant columns
        df = df[['gene_name', 'unstranded', 'Case ID', 'Project ID']]
        df.rename(columns={'unstranded': 'expression'}, inplace=True)
        
        # Pivot the DataFrame to have gene names as columns
        df_pivot = df.pivot_table(index=['Case ID', 'Project ID'], columns='gene_name', values='expression').reset_index()
        
        # Append the pivoted DataFrame to the list
        dfs.append(df_pivot)
    
    # Concatenate all individual DataFrames in the current batch
    batch_df = pd.concat(dfs, axis=0, ignore_index=True)
    
    # Save the batch DataFrame to a CSV file in the cache directory
    batch_df.to_csv(os.path.join(cache_dir, f'batch_{batch_num}.csv'), index=False)

# Combine all batch files into one final unified CSV file
batch_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.startswith('batch_')]

# Initialize an empty list to store batch DataFrames
batch_dfs = []

for batch_file in batch_files:
    # Read the batch DataFrame
    batch_df = pd.read_csv(batch_file)
    # Append the batch DataFrame to the list
    batch_dfs.append(batch_df)

# Concatenate all batch DataFrames into one final DataFrame
final_df = pd.concat(batch_dfs, axis=0, ignore_index=True)

# Save the final DataFrame to a CSV file
final_df.to_csv("/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/unified_gene_expression.csv", index=False)

# Clean up the cache directory
for batch_file in batch_files:
    os.remove(batch_file)
