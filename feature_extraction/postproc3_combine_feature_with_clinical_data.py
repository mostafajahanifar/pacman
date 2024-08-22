import pandas as pd

# Load the csv file (extracted features)
df_csv = pd.read_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/features/_all_combined_single-slide-cases.csv')

# Load the excel file
xls = pd.ExcelFile('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/TCGA-CDR-SupplementalTableS1 (1).xlsx')
df_excel = pd.read_excel(xls, 'TCGA-CDR')  # Load the TCGA-CDR sheet

# Merge the two dataframes on 'bcr_patient_barcode'
merged_df = pd.merge(df_excel, df_csv, on='bcr_patient_barcode')

# Save the merged dataframe to a new csv file
merged_df.to_csv('/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_clinical_merged.csv', index=False)
