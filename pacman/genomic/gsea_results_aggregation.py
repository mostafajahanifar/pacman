import os
import pandas as pd
from openpyxl import Workbook

# Directory containing the cancer types' subdirectories
base_dir = "results_final_all/gene/dseq_results"
csv_file = "gseapy.gene_set.prerank.report.csv"

# Create a writer to write results into an Excel file
output_file = "results_final_all/gene/dseq_results/gsea_filtered_results.xlsx"
writer = pd.ExcelWriter(output_file, engine='openpyxl')

# Function to process each cancer type for GSEA results
def process_gsea_cancer_type(cancer_type, file_path, writer):
    # Read the GSEA results CSV file
    gsea_df = pd.read_csv(file_path)
    gsea_df = gsea_df.drop(columns=["Name"])
    
    # Filter rows based on conditions:
    # 1. Remove rows where "FDR q-val" > 0.01
    filtered_df = gsea_df[gsea_df['FDR q-val'] <= 0.01]
    
    # 2. Keep only rows where the absolute value of "NES" > 1
    filtered_df = filtered_df[filtered_df['NES'].abs() > 1]
    
    # If there are any remaining rows, write to Excel
    if not filtered_df.empty:
        filtered_df.to_excel(writer, sheet_name=cancer_type, index=False)

# Loop through each cancer type's directory
for cancer_type in sorted(os.listdir(base_dir)):
    cancer_dir = os.path.join(base_dir, cancer_type)
    
    if os.path.isdir(cancer_dir):
        file_path = os.path.join(cancer_dir, csv_file)
        
        if os.path.exists(file_path):
            print(cancer_type)
            process_gsea_cancer_type(cancer_type, file_path, writer)

# Save the Excel file
writer.close()

print(f"GSEA results filtered and saved to {output_file}")
