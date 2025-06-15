import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Path to the directory where main event folders will be created
main_directory_path = "./splits/"
# Path to the CSV file with the data table
data_file = "/mnt/gpfs01/lsf-workspace/u2070124/Data/Data/pancancer/tcga_features_final.csv"

# Load the data table
df = pd.read_csv(data_file)

df['type'] = df['type'].replace(['COAD', 'READ'], 'COADREAD')
df['type'] = df['type'].replace(['GBM', 'LGG'], 'GBMLGG')

# Event types to consider
event_types = ["PFI", "OS", "DSS", "DFI"]

# Iterate over each event type
for event_type in event_types:
    # Create a main folder for the current event type
    event_dir = os.path.join(main_directory_path, event_type)
    os.makedirs(event_dir, exist_ok=True)

    # Iterate over unique 'type' in the dataframe
    for experiment_type in df['type'].unique():
        print(event_type, experiment_type)
        experiment_dir = f"{experiment_type}"
        full_experiment_path = os.path.join(event_dir, experiment_dir)
        
        # Filter the dataframe for the current type
        df_type = df[df['type'] == experiment_type]
        

        # Check if the event type column exists in the dataframe
        if event_type not in df_type.columns:
            raise ValueError(f"The event type '{event_type}' is not found in the dataframe columns.")
        
        # Drop rows where the event_type is NaN
        df_type = df_type.dropna(subset=[event_type])
        
        # If the filtered dataframe is empty after dropping NaNs, skip this iteration
        if df_type.empty:
            continue

        if df_type[event_type].sum(axis=0) < 5:
            print(f"Not enough events in {experiment_type} - {event_type}, skip")
            continue
        # Prepare Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Create a new directory for the experiment within the event type folder
        os.makedirs(full_experiment_path, exist_ok=True)

        # Perform the stratified split based on the current event type
        for i, (train_index, val_index) in enumerate(skf.split(df_type, df_type[event_type])):
            train_cases = df_type.iloc[train_index]['bcr_patient_barcode'].tolist()
            val_cases = df_type.iloc[val_index]['bcr_patient_barcode'].tolist()

            # Create a DataFrame for the split
            split_df = pd.DataFrame({
                'train': pd.Series(train_cases).sample(frac=1).reset_index(drop=True),
                'val': pd.Series(val_cases).sample(frac=1).reset_index(drop=True)
            })

            # Save the split DataFrame as a CSV file
            split_df.to_csv(os.path.join(full_experiment_path, f"splits_{i}.csv"), index=False)
