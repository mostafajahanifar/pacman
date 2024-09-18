import pandas as pd

path = '/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/features_final/'

# Load the csv file
df = pd.read_csv(path + '_all_combined.csv')

# Define a function to extract the 'case' from the 'slide'
def extract_case(slide):
    parts = slide.split('-')
    return '-'.join(parts[:3])

def extract_case_brace(slide):
    return slide.split('_')[0]

# Apply the function to the 'slide' column to create the 'case' column
df['bcr_patient_barcode'] = df['slide'].apply(extract_case)

# Sort the DataFrame by 'mit_wsi_count' in descending order and then drop duplicates based on 'case', keeping only the first (highest 'mit_wsi_count')
df = df.sort_values('mit_wsi_count', ascending=False).drop_duplicates('bcr_patient_barcode').sort_index()

# OPTIONAL:: Save the updated DataFrame to a new csv file
df = df[['bcr_patient_barcode'] + [col for col in df.columns if col != 'bcr_patient_barcode']]

# Save the updated DataFrame to a new csv file
df.to_csv(path + '_all_combined_single-slide-cases.csv', index=False)
