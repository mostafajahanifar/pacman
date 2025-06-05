import glob
import pandas as pd
from tqdm import tqdm

path = '/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/features_final/'
# Get a list of all csv files
csv_files = glob.glob(path + '*_header.csv')

# Initialize an empty dataframe
df_combined = pd.DataFrame()

# Loop through the csv files and append to the dataframe
for i, file in tqdm(enumerate(csv_files), total=len(csv_files)):
    df = pd.read_csv(file)
    if i==0:
        df_combined = df.copy()
    else:
        df_combined = pd.concat([df_combined, df], ignore_index=True)

# Write the combined dataframe to a new csv file
df_combined.to_csv(path + '_all_combined.csv', index=False)
